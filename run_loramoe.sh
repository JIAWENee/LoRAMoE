export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}
export CUDA_VISIBLE_DEVICES=0

# export NCCL_NET=IB
# export NCCL_IB_HCA=mlx5_0
# export NCCL_DEBUG=info

lr=0.0002
lora_rank=4
lora_alpha=32
lora_trainable="gate_proj,down_proj,up_proj"
lora_dropout=0.05
lora_nums=8
blc_alpha=0.0
blc_weight=0.0


pretrained_model=/home/LoRAMoE/pre_trained_model/Llama-2-13b-hf
tokenizer_path=/home/LoRAMoE/pre_trained_model/Llama-2-13b-hf
dataset_dir=/home/LoRAMoE/data/tiny_data/train
validation_file=/home/LoRAMoE/data/tiny_data/test.json

per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=1
max_seq_length=1024
output_dir=/home/LoRAMoE/output
exp_name=$(date +"%Y%m%d_%H%M")


# deepspeed_config_file=ds_zero2_no_offload.json
deepspeed_config_file=ds_zero3_offload.json

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 --master_port 29502 \
    run_loramoe.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed 41 \
    --bf16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --save_steps 5000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir}/${exp_name} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_nums ${lora_nums} \
    --blc_alpha ${blc_alpha} \
    --blc_weight ${blc_weight} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype bfloat16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False \
    --flash_attn \
    --overwrite_output_dir \
    &> /home/LoRAMoE/output/log/${exp_name}.log
