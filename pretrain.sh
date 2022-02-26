TRAIN_DIR=./clue_small_wwm_data
OUTPUT_DIR=./wwm_flash_small/
BATCH_SIZE=32
ACCUMULATION=4
LR=1e-4
python run_mlm_wwm.py \
    --do_train \
    --tokenizer_name junnyu/roformer_chinese_char_base \
    --train_dir $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/logs \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION \
    --learning_rate $LR \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps 250000 \
    --warmup_steps 5000 \
    --logging_steps 100 \
    --save_steps 5000 \
    --seed 2022 \
    --max_grad_norm 3.0 \
    --dataloader_num_workers 6 \
    --fp16
