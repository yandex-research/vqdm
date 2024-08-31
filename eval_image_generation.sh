#!/bin/bash
export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3

NUM_GPUS=`python -c "import torch; print(torch.cuda.device_count(), end=\"\")"`

python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1205 --nproc_per_node=$NUM_GPUS eval_image_generation_diffusion.py \
	"stabilityai/stable-diffusion-xl-base-1.0" \
	$QUANTIZATION_MODEL_PATH \
	"eval_prompts/parti-prompts-eval.csv" \
	--scheduler=DDIM --guidance_scale=5 \
  --num_inference_steps 50 \
	--num_samples_per_prompt 4 \
	--bs 1 \
  --save=$DATA_PATH
