# export INSTANCE_DIR="path-to-instance-images"
# export OUTPUT_DIR="path-to-save-model"

# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a painting of sks" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --gradient_checkpointing \
#   --max_train_steps=400

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./inst_dir"
export CLASS_DIR="./cls_dir"
export OUTPUT_DIR="./dreambooth"

accelerate launch examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="sks arxiv paper" \
  --class_prompt="arxiv paper" \
  --enable_xformers_memory_efficient_attention \
  --resolution=768 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --prior_generation_precision="fp16" \
  --set_grads_to_none \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=150 \
  --max_train_steps=800