import argparse
import datetime
import os
import time
from typing import Any, Dict, Sequence  # noqa:F401

import diffusers
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from decomposed_inference import do_inference
from main import add_aqlm_engine_args
from main_text2image import get_pipeline


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
    assert len(all_text) == 128

    num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank() :: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text


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
        "evaluation_prompts",
        type=str,
        help="Path to prompts dataset (newline-separated text file)",
    )
    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=4,
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
        type=float,
        default=1,
        help="Prompt batch size",
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
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="Use teacher model for image generation",
    )

    parser = add_aqlm_engine_args(parser)
    args = parser.parse_args()
    assert args.bs == 1
    assert args.num_samples_per_prompt == 4
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

    if args.teacher:
        pipeline_student = None
    elif os.path.exists(args.quantized_model_path):
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
        raise ValueError("You need to provide valid student path or use `--teacher` for teacher model inference")

    rank_batches, rank_batches_index, all_text = prepare_prompts(args)

    for seed, batch in enumerate(tqdm(rank_batches, unit="batch", disable=(dist.get_rank() != 0))):
        generator = torch.Generator(device=device).manual_seed(seed)
        if pipeline_student:
            print("Generating with a student.")
            images = do_inference(
                pipeline_student,
                prompt=list(batch),
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_samples_per_prompt,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images
        else:
            print("Generating with a teacher.")
            images = do_inference(
                pipeline_teacher,
                prompt=list(batch),
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_samples_per_prompt,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images

        for text_idx, global_idx in enumerate(rank_batches_index[seed]):
            for i in range(args.num_samples_per_prompt):
                idx = args.num_samples_per_prompt * text_idx + i
                images[idx].save(os.path.join(args.save, f"{global_idx}_{i}.jpg"))

    # Done.
    dist.barrier()

    if dist.get_rank() == 0:
        print(f"Overall time: {time.time()-t0:.3f}")
        d = {"caption": all_text}
        df = pd.DataFrame(data=d)
        df.to_csv(os.path.join(args.save, "generated_prompts.csv"))
