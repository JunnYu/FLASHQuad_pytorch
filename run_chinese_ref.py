import argparse
import json
import multiprocessing
import os
import sys
import time
from typing import List

import rjieba
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import BertTokenizerFast


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_chinese(word: str):
    # word like '180' or '身高' or '神'
    for char in word:
        char = ord(char)
        if not _is_chinese_char(char):
            return False
    return True


def get_chinese_word(tokens: List[str]):
    word_set = set()

    for token in tokens:
        chinese_word = len(token) > 1 and is_chinese(token)
        if chinese_word:
            word_set.add(token)
    word_list = list(word_set)
    return word_list


def add_sub_symbol(bert_tokens: List[str], chinese_word_set: set()):
    if not chinese_word_set:
        return bert_tokens
    max_word_len = max([len(w) for w in chinese_word_set])

    bert_word = bert_tokens
    start, end = 0, len(bert_word)
    while start < end:
        single_word = True
        if is_chinese(bert_word[start]):
            l = min(end - start, max_word_len)
            for i in range(l, 1, -1):
                whole_word = "".join(bert_word[start : start + i])
                if whole_word in chinese_word_set:
                    for j in range(start + 1, start + i):
                        bert_word[j] = "##" + bert_word[j]
                    start = start + i
                    single_word = False
                    break
        if single_word:
            start += 1
    return bert_word


block_size = 512


class BlockSizeSplitter:
    def tokenize(self, text):
        tstr = ""
        all_ts = []
        for txt in text.split("\n"):
            if len(tstr) > block_size:
                all_ts.append(tstr)
                tstr = ""
            tstr += txt
        if len(tstr) > 0:
            all_ts.append(tstr)
        return all_ts


def jieba_segmentation_fn():
    def process(line):
        words = rjieba.cut(line)
        return words

    return process


class Converter:
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Converter.tokenizer = BertTokenizerFast.from_pretrained(self.args.model_name)

        # Split document to sentence.
        Converter.splitter = BlockSizeSplitter()
        Converter.segment_func = jieba_segmentation_fn()

        def process(text):
            words = Converter.segment_func(text)
            new_text = "".join(words).replace("\n", "")
            chinese_word = get_chinese_word(words)
            input_tokens = (
                [Converter.tokenizer.cls_token]
                + Converter.tokenizer.tokenize(new_text)
                + [Converter.tokenizer.sep_token]
            )

            input_tokens = add_sub_symbol(input_tokens, chinese_word)
            ref_id = []
            for i, token in enumerate(input_tokens):
                if token[:2] == "##":
                    clean_token = token[2:]
                    # save chinese tokens' pos
                    if len(clean_token) == 1 and _is_chinese_char(ord(clean_token)):
                        ref_id.append(i)

            return ref_id, new_text

        Converter.process = process

    def encode(self, json_line):
        text = json.loads(json_line)[self.args.json_key]
        ref_ids = []
        all_texts = []
        for sentence in Converter.splitter.tokenize(text):
            ref_id, new_text = Converter.process(sentence.strip())
            if len(new_text) < 20:
                continue
            if len(ref_id) > 0 and len(new_text) > 0:
                ref_ids.append(ref_id)
                all_texts.append(new_text)

        return ref_ids, all_texts, len(text.encode("utf-8"))


def main(args):

    file_paths = []
    if os.path.isfile(args.input_path):
        file_paths.append(args.input_path)
    else:
        for root, _, fs in os.walk(args.input_path):
            for f in fs:
                file_paths.append(os.path.join(root, f))
    convert = Converter(args)
    pool = multiprocessing.Pool(args.workers, initializer=convert.initializer)
    step = 0
    total_bytes_processed = 0
    startup_start = time.time()
    with open("data/refids.txt", "w+", encoding="utf8") as w1:
        with open("data/reftext.txt", "w+", encoding="utf8") as w2:
            for file_path in tqdm(file_paths):
                if file_path.endswith(".jsonl"):
                    text = open(file_path, "r", encoding="utf-8")
                else:
                    print("Unexpected data format, skiped %s" % file_path)
                    continue

                encoded_docs = pool.imap(convert.encode, text, 256)
                print("Processing %s" % file_path)
                for rid, alltxt, bytes_processed in encoded_docs:
                    step += 1
                    total_bytes_processed += bytes_processed
                    if len(rid) == 0:
                        continue

                    for sentence in rid:
                        sentence_len = len(sentence)
                        if sentence_len == 0:
                            continue
                        w1.write(str(sentence) + "\n")
                    for txt in alltxt:
                        txt_len = len(txt)
                        if txt_len == 0:
                            continue
                        w2.write(txt + "\n")

                    if step % args.log_interval == 0:
                        current = time.time()
                        elapsed = current - startup_start
                        mbs = total_bytes_processed / elapsed / 1024 / 1024
                        print(
                            f"Processed {step} documents",
                            f"({step/elapsed:.2f} docs/s, {mbs:.4f} MB/s).",
                            file=sys.stderr,
                        )
            pool.close()
            print("Saving tokens to files...")

    # concatenate_datasets
    print("concatenate_datasets...")
    reftext = load_dataset("text", data_files="data/reftext.txt")["train"]
    refids = load_dataset("text", data_files="data/refids.txt")["train"]
    refids = refids.rename_column("text", "chinese_ref")
    refids = refids.map(lambda example: {"chinese_ref": eval(example["chinese_ref"])})
    concat_ds = concatenate_datasets([reftext, refids], axis=1)
    concat_ds.save_to_disk("./clue_small_wwm_data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="junnyu/roformer_chinese_char_base",
        help="What model to use.",
    )

    group = parser.add_argument_group(title="data input/output")
    group.add_argument(
        "--input_path",
        type=str,
        default="data/clue_corpus_small_14g.jsonl",
        help="Path to input JSON files.",
    )

    group.add_argument(
        "--json_key",
        type=str,
        default="text",
        help="For JSON format. Space separate listed of keys to extract from json",
    )

    group = parser.add_argument_group(title="common config")

    group.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    group.add_argument(
        "--workers", type=int, default=12, help="Number of worker processes to launch"
    )

    args = parser.parse_args()
    main(args)
