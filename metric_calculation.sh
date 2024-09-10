#!/bin/bash
export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
NUM_GPUS=`python -c "import torch; print(torch.cuda.device_count(), end=\"\")"`

python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1205 --nproc_per_node=4 eval_metric_calculation_diffusion.py \
  "stabilityai/stable-diffusion-xl-base-1.0" \
	$QUANTIZATION_MODEL_PATH \
	"evaluation_prompts=eval_prompts/coco.csv" \
	 --scheduler=DDIM --guidance_scale=5 \
  --clip_model_name_or_path=${INPUT_PATH}/CLIP-ViT-H-14-laion2B-s32B-b79K --pickscore_model_name_or_path=${INPUT_PATH}/PickScore_v1 \
  --num_inference_steps=50 --coco_ref_stats_path="stats/fid_stats_mscoco512_val.npz" --inception_path="stats/pt_inception-2015-12-05-6726825d.pth" \
	--num_samples_per_prompt 1 \
	--bs 1 \
  --max_count=5000 \
  --save=$DATA_PATH \
  --wandb
