import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    MultipleChoiceModelOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary

from transformers.utils import logging

logger = logging.get_logger(__name__)

###################################################copied from https://github.com/lucidrains/FLASH-pytorch
import math
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


# scalenorm


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# T5 relative positional bias


class T5RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j 1 -> i j")
        return bias * self.scale


# class


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, dim))
        self.bias = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        out = einsum("... d, h d -> ... h d", x, self.weight) + self.bias
        return out.unbind(dim=-2)


# FLASH


class FLASH(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=2.0,
        causal=False,
        dropout=0.0,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=False
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # positional embeddings

        self.rotary_pos_emb = rotary_pos_emb
        self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal=causal)

        # norm

        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout)

        # projections

        self.to_hidden = nn.Sequential(nn.Linear(dim, hidden_dim * 2), nn.SiLU())

        self.to_qk = nn.Sequential(nn.Linear(dim, query_key_dim), nn.SiLU())

        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, attention_mask=None, output_attentions=False):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """
        mask = attention_mask
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # prenorm

        normed_x = self.norm(x)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.0)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        # initial projections

        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        # offset and scale

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        # mask out linear attention keys

        if exists(mask):
            mask = mask.bool()
            lin_k = lin_k.masked_fill(~mask[..., None], 0.0)

        # rotate queries and keys

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys,
                (quad_q, lin_q, quad_k, lin_k),
            )

        # padding for groups

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.0),
                (quad_q, quad_k, lin_q, lin_k, v),
            )

            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v = map(
            lambda t: rearrange(t, "b (g n) d -> b g n d", n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v),
        )

        if exists(mask):
            mask = rearrange(mask, "b (g j) -> b g 1 j", j=g)

        # calculate quadratic attention output

        sim = einsum("... i d, ... j d -> ... i j", quad_q, quad_k) / g

        sim = sim + self.rel_pos_bias(sim)

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.0)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.0)

        quad_out = einsum("... i j, ... j d -> ... i d", attn, v)

        # calculate linear attention output

        if self.causal:
            lin_kv = einsum("b g n d, b g n e -> b g d e", lin_k, v) / g

            # exclusive cumulative sum along group dimension

            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.0)

            lin_out = einsum("b g d e, b g n d -> b g n e", lin_kv, lin_q)
        else:
            lin_kv = einsum("b g n d, b g n e -> b d e", lin_k, v) / n
            lin_out = einsum("b g n d, b d e -> b g n e", lin_q, lin_kv)

        # fold back groups into full sequence, and excise out padding

        quad_attn_out, lin_attn_out = map(
            lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n], (quad_out, lin_out)
        )

        # gate

        out = gate * (quad_attn_out + lin_attn_out)

        # projection out and residual
        out = self.to_out(out) + x
        return (out, attn) if output_attentions else (out,)


###################################################


class FLASHConfig(PretrainedConfig):
    model_type = "flash"

    def __init__(
        self,
        vocab_size=12000,
        hidden_size=768,
        num_hidden_layers=12,  # base
        max_position_embeddings=512,
        group_size=256,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        expansion_factor=2,
        query_key_dim=128,
        norm_type="scalenorm",
        gradient_checkpointing=False,
        dropout=0.0,
        classifier_dropout=0.1,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.group_size = group_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.expansion_factor = expansion_factor
        self.query_key_dim = query_key_dim
        self.norm_type = norm_type
        self.dropout = dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.classifier_dropout = classifier_dropout


class FLASHPreTrainedModel(PreTrainedModel):
    config_class = FLASHConfig
    base_model_prefix = "flash"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FLASHLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states):
        return self.decoder(self.LayerNorm(hidden_states))


class FLASHOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = FLASHLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class FLASHEmbeddings(nn.Module):
    """Construct the embeddings from word, position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer(
            "scaledsin_embeddings", self.get_scaledsin(), persistent=False
        )

    def get_scaledsin(self):
        """Create sinusoidal position embedding with a scaling factor."""
        seqlen, hidden_size = (
            self.config.max_position_embeddings,
            self.config.hidden_size,
        )
        pos = torch.arange(seqlen, dtype=torch.float32)
        half_d = hidden_size // 2

        freq_seq = -torch.arange(half_d, dtype=torch.float32) / float(half_d)
        inv_freq = 10000 ** freq_seq
        sinusoid = torch.einsum("s,d->sd", pos, inv_freq)
        scaledsin = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return scaledsin

    def forward(self, input_ids=None, position_ids=None):
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.scaledsin_embeddings[position_ids] * self.scale
        embeddings = inputs_embeds + position_embeddings
        return embeddings


class FLASHEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.norm_type == "scalenorm":
            norm_klass = ScaleNorm
        elif config.norm_type == "layernorm":
            norm_klass = nn.LayerNorm
        rotary_pos_emb = RotaryEmbedding(dim=min(32, config.query_key_dim))
        self.layers = nn.ModuleList(
            [
                FLASH(
                    dim=config.hidden_size,
                    group_size=config.group_size,
                    query_key_dim=config.query_key_dim,
                    expansion_factor=config.expansion_factor,
                    causal=False,
                    dropout=config.dropout,
                    rotary_pos_emb=rotary_pos_emb,
                    norm_klass=norm_klass,
                    shift_tokens=False,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class FLASHModel(FLASHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = FLASHEmbeddings(config)
        self.encoder = FLASHEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FLASHForMaskedLM(FLASHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.flash = FLASHModel(config)
        self.cls = FLASHOnlyMLMHead(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.flash(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FLASHForSequenceClassification(FLASHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.flash = FLASHModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.flash(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = self.sequence_summary(outputs[0])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FLASHForMultipleChoice(FLASHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.flash = FLASHModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        num_choices = input_ids.shape[1]
        input_ids = (
            input_ids.reshape(-1, input_ids.size(-1)) if input_ids is not None else None
        )
        attention_mask = (
            attention_mask.reshape(-1, attention_mask.size(-1))
            if attention_mask is not None
            else None
        )
        position_ids = (
            position_ids.reshape(-1, position_ids.size(-1))
            if position_ids is not None
            else None
        )

        outputs = self.flash(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.sequence_summary(outputs[0])
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FLASHForTokenClassification(FLASHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.flash = FLASHModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.dropout
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.flash(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FLASHForQuestionAnswering(FLASHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.flash = FLASHModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.flash(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if start_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
