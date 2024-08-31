import argparse
import copy
import glob
import logging
import os
import random
import sys

import diffusers
import torch

from decomposed_inference import do_inference
from latents_dataset import gather_unet_latents
from main import add_aqlm_engine_args
from main_text2image import (
    do_finetuning,
    draw_grid,
    get_datasets_for_finetuning,
    get_pipeline,
    remove_snapshots,
    testing_model,
)
from src.aq import QuantizedConv2D, QuantizedLinear
from src.datautils import get_diffusion_prompts, set_seed

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="[%(asctime)s][ft] %(message)s")
logger = logging.getLogger()

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

if __name__ == "__main__":
    ##load pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="path to  model to load, as in AutoPipelineForText2Image.from_pretrained()",
    )
    parser.add_argument(
        "quantized_model_path",
        type=str,
        help="path to quantized diffusion model to load, as in AutoPipelineForText2Image.from_pretrained()",
    )
    parser.add_argument(
        "calibration_prompts",
        type=str,
        help="Path to calibration prompts dataset (newline-separated text file)",
    )
    parser.add_argument(
        "evaluation_prompts",
        type=str,
        help="Path to evaluation prompts dataset (newline-separated text file)",
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
        "--finetune_nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--evaluation_nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--visualize_nsamples",
        type=int,
        default=8,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="number of inference steps used for calibration and evaluation",
    )
    parser.add_argument(
        "--min_channels",
        type=int,
        default=16,
        help="min channels to perform quantization",
    )
    parser.add_argument(
        "--finetune_method",
        type=str,
        default="teacher",
        choices=["teacher", "student", "real"],
        help="On which latent code will be finetuned."
        "teacher - latent code obtained with unquantized pipeline,"
        "student - latent code obtained with quantized pipeline, "
        "real - latent code obtained from noised real latent data(not implemented)",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")

    parser = add_aqlm_engine_args(parser)

    args = parser.parse_args()
    args.code_dtype = getattr(torch, args.code_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.devices is None:
        args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        args.devices = [torch.device(device_str) for device_str in args.devices]
    assert all(isinstance(device, torch.device) for device in args.devices)
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "AQDM")
            + f"_finetune_nsamples_{args.finetune_nsamples}"
            + f"_finetune_lr_{args.finetune_lr}"
            + f"_finetune_relative_mse_tolerance_{args.finetune_relative_mse_tolerance}"
            + f"_{len(args.devices)}gpus"
        )
        args.group_size = args.in_group_size * args.out_group_size

        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )

    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info("Generation seed:", seed)

    set_seed(seed)

    # load teacher and student
    pipeline_teacher = get_pipeline(
        model_path=args.model_path, scheduler_name=args.scheduler, dtype=args.dtype, device=device
    )

    do_inference(
        pipeline_teacher,
        prompt="tree",
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    pipeline_student = get_pipeline(
        model_path=args.model_path, scheduler_name=args.scheduler, dtype=args.dtype, device=device
    )

    pickle_files_paths = []
    if args.resume and args.save:
        os.makedirs(args.save, exist_ok=True)
        pickle_files_paths = [f for f in glob.glob(args.save + "/*.pickle")]

    if pickle_files_paths:
        pipeline_student.unet = torch.load(pickle_files_paths[0], map_location=device).to(device)
        start_epoch = int(pickle_files_paths[0].split("_")[-1].split(".")[0])
        print(f"Continue finetuning form {start_epoch=}")
    else:
        loaded_module = torch.load(args.quantized_model_path, map_location=device).to(device)

        if isinstance(loaded_module, diffusers.StableDiffusionXLPipeline):
            pipeline_student.unet = loaded_module.unet
        else:
            pipeline_student.unet = loaded_module
        start_epoch = 0

    calibration_prompts = get_diffusion_prompts(
        args.calibration_prompts, nsamples=args.finetune_nsamples * args.finetune_max_epochs, seed=seed, eval_mode=False
    )
    evaluation_prompts = get_diffusion_prompts(
        args.evaluation_prompts, nsamples=args.evaluation_nsamples, seed=seed, eval_mode=True
    )
    # cast codes to int32
    for module in pipeline_student.unet.modules():
        if isinstance(module, QuantizedConv2D) or isinstance(module, QuantizedLinear):
            module.quantized_weight.set_codes(
                torch.nn.Parameter(module.quantized_weight.get_codes().to(args.code_dtype), requires_grad=False)
            )
        if "Attention" in repr(type(module)) or "FeedForward" in repr(type(module)):
            for child in module.modules():
                if isinstance(module, QuantizedConv2D) or isinstance(module, QuantizedLinear):
                    # not working
                    module.quantized_weight.set_codes(
                        torch.nn.Parameter(module.quantized_weight.get_codes().to(args.code_dtype), requires_grad=False)
                    )

    max_epoch = args.finetune_max_epochs
    args.finetune_max_epochs = 1
    for epoch in range(start_epoch, max_epoch):
        # getting datasets
        datasets = []
        if args.finetune_method == "teacher":
            datasets = get_datasets_for_finetuning(
                pipeline_teacher,
                calibration_prompts[epoch * args.finetune_nsamples : (epoch + 1) * args.finetune_nsamples],
                args,
            )

        pipeline_student.unet = do_finetuning(
            pipeline_student,
            pipeline_teacher,
            calibration_prompts[epoch * args.finetune_nsamples : (epoch + 1) * args.finetune_nsamples],
            datasets,
            args,
        )
        if args.wandb:
            testing_model(
                pipeline_student,
                pipeline_teacher,
                evaluation_prompts,
                args=args,
                nsamples=args.evaluation_nsamples,
                seed=seed,
                step=epoch,
            )
        if args.resume and args.save:
            # Removing previous snapshots
            os.makedirs(args.save, exist_ok=True)
            remove_snapshots(args)
            torch.save(pipeline_student.unet, os.path.join(args.save, f"quantized_unet_{epoch+1}.pickle"))
            if args.on_save:
                exec(args.on_save)

    # monkey patching - just please don't ask
    pipeline_student.vae = pipeline_teacher.vae

    grid = draw_grid(
        pipeline_student=pipeline_student,
        pipeline_teacher=pipeline_teacher,
        prompts=evaluation_prompts,
        args=args,
        seed=seed,
        num_images=args.visualize_nsamples,
    )
    if args.wandb:
        images = wandb.Image(grid, caption="Left: Student, Right: Teacher")
        wandb.log({"Final_examples": images})
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        grid.save(os.path.join(args.save, "image_pairs.png"))
        logger.info("Images saved to", os.path.join(args.save, "image_pairs.png"))

    # monkey patching over
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        remove_snapshots(args)
        torch.save(pipeline_student.unet, os.path.join(args.save, "quantized_unet.pickle"))
        if args.on_save:
            exec(args.on_save)
