import argparse
import datetime
import os
import random
import time
from typing import Any, Dict, Sequence

import diffusers
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.distributed as dist
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from decomposed_inference import do_inference
from main import add_aqlm_engine_args
from main_text2image import get_pipeline
from src.fid_score_in_memory import calculate_fid

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def dist_init():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"

    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(0, 3600))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def prepare_prompts(args):
    df = pd.read_csv(args.evaluation_prompts)
    all_text = list(df["captions"])
    all_text = all_text[: args.max_count]

    num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank() :: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text


@torch.no_grad()
def distributed_sampling(pipeline, device, weight_dtype, args):

    pipeline.set_progress_bar_config(disable=True)

    pipeline = pipeline.to(device)
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    rank_batches, rank_batches_index, all_prompts = prepare_prompts(args)

    local_images = []
    local_text_idxs = []
    for cnt, mini_batch in enumerate(tqdm(rank_batches, unit="batch", disable=(dist.get_rank() != 0))):
        images = do_inference(
            pipeline,
            prompt=list(mini_batch),
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.num_samples_per_prompt,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images

        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.tensor(np.array(images[text_idx]))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)

    local_images = torch.stack(local_images).cuda()
    local_text_idxs = torch.tensor(local_text_idxs).cuda()

    gathered_images = [torch.zeros_like(local_images) for _ in range(dist.get_world_size())]
    gathered_text_idxs = [torch.zeros_like(local_text_idxs) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_images, local_images)  # gather not supported with NCCL
    dist.all_gather(gathered_text_idxs, local_text_idxs)

    images, prompts = [], []
    if dist.get_rank() == 0:
        gathered_images = np.concatenate([images.cpu().numpy() for images in gathered_images], axis=0)
        gathered_text_idxs = np.concatenate([text_idxs.cpu().numpy() for text_idxs in gathered_text_idxs], axis=0)
        for image, global_idx in zip(gathered_images, gathered_text_idxs):
            images.append(ToPILImage()(image))
            prompts.append(all_prompts[global_idx])
    # Done.
    dist.barrier()
    return images, prompts


@torch.no_grad()
def calc_pick_and_clip_scores(model, image_inputs, text_inputs, batch_size=50):
    assert len(image_inputs) == len(text_inputs)

    scores = torch.zeros(len(text_inputs))
    for i in range(0, len(text_inputs), batch_size):
        image_batch = image_inputs[i : i + batch_size]
        text_batch = text_inputs[i : i + batch_size]
        # embed
        with torch.cuda.amp.autocast():
            image_embs = model.get_image_features(image_batch)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.cuda.amp.autocast():
            text_embs = model.get_text_features(text_batch)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores[i : i + batch_size] = (text_embs * image_embs).sum(-1)  # model.logit_scale.exp() *
    return scores.cpu()


def calculate_scores(args, images, prompts, device="cuda"):
    processor = AutoProcessor.from_pretrained(args.clip_model_name_or_path)
    clip_model = AutoModel.from_pretrained(args.clip_model_name_or_path).eval().to(device)
    pickscore_model = AutoModel.from_pretrained(args.pickscore_model_name_or_path).eval().to(device)

    image_inputs = processor(images=images, return_tensors="pt",)[
        "pixel_values"
    ].to(device)

    text_inputs = processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt",)[
        "input_ids"
    ].to(device)

    print("Evaluating PickScore...")
    pick_score = calc_pick_and_clip_scores(pickscore_model, image_inputs, text_inputs).mean()
    print("Evaluating CLIP ViT-H-14 score...")
    clip_score = calc_pick_and_clip_scores(clip_model, image_inputs, text_inputs).mean()
    print("Evaluating FID score...")
    fid_score = calculate_fid(
        images, args.coco_ref_stats_path, inception_path=args.inception_path
    )  # https://github.com/yandex-research/lcm/tree/9886452e69931b2520a8ec43540b50acef243ca4/stats
    return pick_score, clip_score, fid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="path to model to load, as in AutoPipelineForText2Image.from_pretrained()",
    )
    parser.add_argument(
        "quantized_model_path",
        type=str,
        help="path to quantized diffusion model to load, as in AutoPipelineForText2Image.from_pretrained()",
    )
    parser.add_argument(
        "--evaluation_prompts",
        default="eval_prompts/coco.csv",
        type=str,
        help="Path to prompts dataset (newline-separated text file)",
    )
    parser.add_argument(
        "--clip_model_name_or_path",
        default="${INPUT_PATH}/CLIP-ViT-H-14-laion2B-s32B-b79K",
        type=str,
        help="path to clip model to load, as in AutoModel.from_pretrained",
    )

    parser.add_argument(
        "--pickscore_model_name_or_path",
        default="${INPUT_PATH}/PickScore_v1",
        type=str,
        help="path to pickscore model to load, as in AutoModel.from_pretrained",
    )
    parser.add_argument(
        "--coco_ref_stats_path",
        default="stats/fid_stats_mscoco512_val.npz",
        type=str,
        help="Path to reference stats from coco",
    )
    parser.add_argument(
        "--inception_path",
        default="stats/pt_inception-2015-12-05-6726825d.pth",
        type=str,
        help="Path to inception reference stats ",
    )

    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="number of inference steps used for calibration and evaluation",
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=1,
        help="Prompt batch size",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=5000,
        help="Prompt count to eval on ",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        required=True,
        help="Scheduler type from ['DDPM', 'DDIM', 'Heun', 'DPMSolver', 'ODE']. Use 'default' to load from config",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        required=True,
        help="Guidance scale as defined in [Classifier-Free Diffusion Guidance]",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")

    parser = add_aqlm_engine_args(parser)
    args = parser.parse_args()
    assert args.num_samples_per_prompt == 1
    dist_init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)

    # TODO
    dtype = torch.float16
    assert dtype == torch.float16

    if dist.get_rank() == 0:
        t0 = time.time()
        os.makedirs(args.save, exist_ok=True)

    # load teacher and student
    pipeline_teacher = get_pipeline(
        model_path=args.model_path, scheduler_name=args.scheduler, dtype=dtype, device=device
    )

    if args.wandb and dist.get_rank() == 0:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = os.environ.get("WANDB_NAME", "AQDM")

        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )

    if os.path.exists(args.quantized_model_path):
        pipeline_student = get_pipeline(
            model_path=args.model_path, scheduler_name=args.scheduler, dtype=dtype, device=device
        )

        # To be sure that scheduler is DDIMScheduler
        loaded_module = torch.load(args.quantized_model_path, map_location=device).to(device)

        if isinstance(loaded_module, diffusers.StableDiffusionXLPipeline):
            pipeline_student.unet = loaded_module.unet
        else:
            pipeline_student.unet = loaded_module
    else:
        pipeline_student = None

    if pipeline_student:
        print("Generating with a student.")
        images, prompts = distributed_sampling(pipeline_student, device, dtype, args)
    else:
        print("Generating with a teacher.")
        images, prompts = distributed_sampling(pipeline_teacher, device, dtype, args)

    if dist.get_rank() == 0:
        pick_score, clip_score, fid_score = calculate_scores(args, images, prompts, device="cuda")
        if args.wandb:
            wandb.log({"pick_score": pick_score, "clip_score": clip_score, "fid_score": fid_score})
        print(f"{pick_score}", f"{clip_score}", f"{fid_score}")
