#!/bin/bash
export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
NUM_GPUS=`python -c "import torch; print(torch.cuda.device_count(), end=\"\")"`

python3 finetune_diffusion.py \
	"stabilityai/stable-diffusion-xl-base-1.0" \
	$INPUT_PATH/quantized_unet.pickle  pickscore pickscore --finetune_nsamples=512 \
  --evaluation_nsamples=32 --num_inference_steps=50 --scheduler=DDIM --guidance_scale=5 \
  --dtype float16 --finetune_lr=0.000001 --finetune_relative_mse_tolerance=0.001 \
  --finetune_batch_size=64 --local_batch_size=1 \
  --finetune_max_epoch=4 --print_frequency=1 \
  --finetune_method=teacher --group_channels --finetune_adam_beta2=0.999 \
  --save $DATA_PATH \
  --resume \
  --wandb
