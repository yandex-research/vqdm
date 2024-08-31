import argparse
import glob
import logging
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, Collection, Dict, Optional, Sequence, Tuple, Union

import diffusers
import lpips
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from diffusers.utils import make_image_grid
from torch.nn.modules.conv import _ConvNd
from tqdm import tqdm, trange

from mixture_unet import MixtureOfUnets

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="[%(asctime)s][%(name)s] %(message)s")
logger = logging.getLogger("t2i")

from aq_engine import AQEngine
from decomposed_inference import do_inference
from latents_dataset import gather_unet_latents
from main import _LayerWrapperThatAccumulatesXTX, add_aqlm_engine_args  # TODO extract
from src.aq import QuantizedConv2D, QuantizedLinear
from src.datautils import get_diffusion_prompts, set_seed
from src.finetune_unet import finetune_unet

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

# quantizes all linear and convolutional layers (time embedding is skipped)
DEFAULT_LAYER_REGEX = "(down|mid|up)_blocks?.*(to_(q|k|v|out.0)|net\.(0\.proj|2)|conv(\d+|_shortcut)?|proj_(in|out))$"
# quantizes all linear and convolutional layers without exception
ALL_LAYER_REGEX = ".*"
# quantizes only linear layers
LINEAR_LAYER_ONLY_REGEX = "(down|mid|up)_blocks?.*(to_(q|k|v|out.0)|net\.(0\.proj|2)|proj_(in|out))$"
# quantizes only convolutional layers
CONV_LAYER_ONLY_REGEX = "(down|mid|up)_blocks?.*conv(\d+|_shortcut)?$"


def determine_layer_order(
    pipeline: diffusers.StableDiffusionXLPipeline,
    layer_types: Collection[type] = (nn.Conv2d,),
    min_channels: int = 16,
    **kwargs,
) -> OrderedDict[str, nn.Module]:
    """
    Run a diffuser pipeline and determine in which order does it apply layers of certain type or types
    :param layer_types: return only layers of this type or types (default = any nn.Conv2d layers)
    :param min_channels: for linear and convolution layers, ignore any layers whose in/out dimension is lower than this
    :param kwargs: forwarded to inference code
    :returns: an ordered dict whose items are (layer name, layer) in the same order as they are called during forward

    """
    layer_order = OrderedDict()

    def log_name(name):
        def tmp(layer, inp, out):
            if name not in layer_order:
                layer_order[name] = layer

        return tmp

    handles = []
    for i, layer in pipeline.unet.named_modules():
        if isinstance(layer, layer_types):
            if isinstance(layer, _ConvNd) and min(layer.in_channels, layer.out_channels) < min_channels:
                continue
            elif isinstance(layer, nn.Linear) and min(layer.in_features, layer.out_features) < min_channels:
                continue
            handles.append(layer.register_forward_hook(log_name(i)))

    torch.manual_seed(0)
    _ = do_inference(pipeline, output_type="latent", **kwargs)

    for handle in handles:
        handle.remove()
    assert len(layer_order) > 0
    return layer_order


def init_aq_engines(
    pipeline: diffusers.StableDiffusionXLPipeline,
    prompts: Sequence[str],
    subset: Dict[str, nn.Module],
    **kwargs: Dict[str, Any],
) -> Dict[str, AQEngine]:
    aq_handlers: Dict[str, AQEngine] = {}
    for sublayer_name in subset:
        aq_handlers[sublayer_name] = AQEngine(subset[sublayer_name])

    found_module = {module: False for module in subset.values()}
    wrapped_layer_to_hander = {aq_handler.layer: aq_handler for aq_handler in aq_handlers.values()}
    for module in list(pipeline.unet.modules()):
        for child_name, child in list(module.named_children()):
            if child in wrapped_layer_to_hander:
                setattr(module, child_name, _LayerWrapperThatAccumulatesXTX(child, wrapped_layer_to_hander[child]))
                found_module[child] = True
    assert all(found_module.values())

    with torch.no_grad():
        for start in range(0, len(prompts), args.xtx_batch_size):
            end = min(start + args.xtx_batch_size, len(prompts))
            do_inference(pipeline, prompt=prompts[start:end], output_type="latent", **kwargs)

    # remove wrappers
    for module in list(pipeline.unet.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, _LayerWrapperThatAccumulatesXTX):
                setattr(module, child_name, child.wrapped_layer)
    return aq_handlers


@torch.no_grad()
def init_aq_engines_parallel(
    devices: Sequence[torch.device],
    pipeline: diffusers.StableDiffusionXLPipeline,
    student_unet_index: Optional[int],
    prompts: Sequence[str],
    subset: Dict[str, nn.Module],
    **kwargs: Dict[str, Any],
):
    """Parallel version of init_aq_engines; works on lists of input/output tensors"""
    pipeline_replicas = replicate_text2image_pipeline(
        pipeline, devices=devices, detach=True, reuse_first_replica=True, add_dummy_parameter=True
    )
    assert (
        pipeline_replicas[0] is pipeline
    )  # this ensures that aq_handlers returned by 0-th replica operate on the main layer
    funcs_by_device = [init_aq_engines for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    assert len(prompts) % len(devices) == 0
    per_device_prompts_count = (len(prompts) - 1) // len(devices) + 1
    for i in range(len(devices)):
        if i == 0:
            replica_subset = subset
        else:
            replica_unet = pipeline_replicas[i].unet
            assert isinstance(replica_unet, MixtureOfUnets) == (student_unet_index is not None)
            if student_unet_index is not None:
                replica_unet: nn.Module = replica_unet.models[student_unet_index]
            replica_modules_by_name = dict(replica_unet.named_modules())
            replica_subset = {name: replica_modules_by_name[name] for name in subset}
        inputs_by_device.append(
            (
                pipeline_replicas[i],
                prompts[i * per_device_prompts_count : (i + 1) * per_device_prompts_count],
                replica_subset,
            )
        )
        kwargs_by_device.append(
            {k: (v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        )
    aq_handles_by_device: Sequence[Dict[str, AQEngine]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    aq_handlers = aq_handles_by_device[0]
    for key, aq_handler in aq_handlers.items():
        replica_handlers = [device_aq_handlers[key] for device_aq_handlers in aq_handles_by_device]
        replica_nsamples = [replica_handler.nsamples for replica_handler in replica_handlers]
        total_nsamples = sum(replica_nsamples)
        aq_handler.XTX = sum(
            (replica_handlers[i].XTX * (replica_nsamples[i] / total_nsamples)).to(devices[0], non_blocking=True)
            for i in range(len(devices))
        )
        aq_handler.nsamples = total_nsamples
    return aq_handlers


def replicate_text2image_pipeline(
    pipeline: diffusers.StableDiffusionXLPipeline,
    devices: Sequence[torch.device],
    detach: bool,
    reuse_first_replica: bool = False,
    add_dummy_parameter: bool = False,
) -> Sequence[diffusers.StableDiffusionXLPipeline]:
    """Similar to torch.nn.parallel.replicate, but for diffusion pipelines"""

    module_dict_for_replication = nn.ModuleDict(
        {k: v for k, v in pipeline.components.items() if isinstance(v, nn.Module)}
    )
    component_replicas = nn.parallel.replicate(module_dict_for_replication, devices=devices, detach=detach)

    replicas = []
    if reuse_first_replica:
        replicas.append(pipeline)

    for replica_index in range(int(reuse_first_replica), len(devices)):
        replica_memo = {}
        for key in pipeline.components:
            original_module = pipeline.components[key]
            if key in module_dict_for_replication:
                replica_module = component_replicas[replica_index][key]
                replica_memo[id(original_module)] = replica_module  # replace with pre-replicated module
            elif key.startswith("tokenizer"):
                replica_memo[id(original_module)] = original_module  # do not copy
            else:
                pass  # replicate scheduler since it is (sometimes) stateful

        replica = deepcopy(pipeline, memo=replica_memo)
        replicas.append(replica)

    if add_dummy_parameter:
        dtype = next(iter(pipeline.vae.parameters())).dtype  # unsafe for replicas
        for i, replica in enumerate(replicas):
            dummy_parameter = nn.Parameter(torch.empty(0, dtype=dtype, device=devices[i]), requires_grad=False)
            for key in module_dict_for_replication.keys():
                assert isinstance(replica.components[key], nn.Module)
                for replica_module in replica.components[key].modules():
                    prev_value = getattr(replica_module, "_dummy_parameter", None)
                    assert prev_value is None or prev_value.numel() == 0
                    replica_module._dummy_parameter = dummy_parameter
    return replicas


def quantize_unet_layers_inplace_(
    pipeline: diffusers.StableDiffusionXLPipeline,
    prompts: Sequence[str],
    subset: Dict[str, nn.Module],
    teacher_unet: nn.Module,
    args: argparse.Namespace,
    subset_step=None,
    **kwargs,
):
    """
    Quantize a subset of conv or linear layers within pipline and replace original layers with quantized ones
    :param pipeline: a diffusion pipeline to be quantized
    :param prompts: texts used as calibration data
    :param subset: which layers are to be quantized
    :param args: AQLM parameters for these layers
    :param kwargs: forwarded to diffusion forward pass
    :param subset_step: current subset step
    """
    if len(subset) == 0:
        logger.info("Subset is empty or already quantized.")
        return pipeline
    ### 1. accumulate XTX matrices from calibration data ###
    # wrap all quantized sub-layers with a wrapper that accumulates inputs on forward
    # note: the code below uses wrappers instead of hooks because hooks cause bugs in multi-gpu code
    logger.info(f"Initializing AQ engines. Will run {len(prompts) // args.xtx_batch_size} inference loops")
    original_unet = pipeline.unet
    student_unet_index = None
    if args.max_timestep != pipeline.scheduler.config["num_train_timesteps"] or args.min_timestep != 0:
        intervals = []
        if args.max_timestep != pipeline.scheduler.config["num_train_timesteps"]:
            intervals.append(((pipeline.scheduler.config["num_train_timesteps"], args.max_timestep), teacher_unet))
        intervals.append(((args.max_timestep, args.min_timestep), original_unet))
        if args.min_timestep != 0:
            intervals.append(((args.min_timestep, 0), teacher_unet))
        pipeline.unet = MixtureOfUnets(OrderedDict(intervals))
        student_unet_index = list(pipeline.unet.models).index(original_unet)
    if len(args.devices) == 1:
        aq_handlers = init_aq_engines(pipeline, prompts, subset, **kwargs)
    else:
        aq_handlers = init_aq_engines_parallel(args.devices, pipeline, student_unet_index, prompts, subset, **kwargs)
    pipeline.unet = original_unet
    logger.info(f"init_aq_engines complete in {time.perf_counter() - subset_start_time:.1f}s.")
    ### 2. quantize each layer using accumulated XTX ###

    for sublayer_name in aq_handlers.keys():
        sublayer_start_time = time.perf_counter()
        logger.info("-" * 180)
        params_count = aq_handlers[sublayer_name].layer.weight.numel()
        logger.info(f"Quantizing module `{sublayer_name}`  params count:{params_count:,}  from subset {subset_step}")
        logger.info(f"module params: {str(aq_handlers[sublayer_name].layer)}")

        quantized_weight = aq_handlers[sublayer_name].quantize(args=args, verbose=True)
        original_layer = aq_handlers[sublayer_name].layer
        assert aq_handlers[sublayer_name].layer.weight in set(
            original_layer.parameters()
        )  # test that this is not a replica
        #         assert isinstance(original_layer, nn.Conv2d)

        with torch.no_grad():
            if isinstance(original_layer, nn.Conv2d):
                quantized_layer = QuantizedConv2D.from_conv(
                    quantized_weight, conv2d=original_layer, group_channels=args.group_channels
                )
                quantized_layer = quantized_layer.to(original_layer.weight.dtype)
            else:
                assert isinstance(original_layer, nn.Linear)
                quantized_layer = QuantizedLinear(quantized_weight, aq_handlers[sublayer_name].layer.bias)

            if args.use_checkpointing:
                quantized_layer.use_checkpoint = True
                logger.info("ENABLED CHECKPOINTING FOR", sublayer_name)

            for submodule in pipeline.unet.modules():
                for child_name, child_module in submodule.named_children():
                    if child_module is aq_handlers[sublayer_name].layer:
                        setattr(submodule, child_name, quantized_layer)
                        found_original = True  # note: do not break to handle tied layers
            assert found_original, f"could not find {sublayer_name}"
        elapsed = time.perf_counter() - sublayer_start_time
        logger.info(f"module `{sublayer_name}` quantized in {elapsed:,.0f}s.")
    del aq_handlers
    return pipeline


def generate_image_pairs(
    prompt: Union[str, Sequence[str]],
    seed: int,
    left_model: diffusers.StableDiffusionXLPipeline,
    right_model: diffusers.StableDiffusionXLPipeline,
    **kwargs,
) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
    """Generate pairs of images from two models using the same random seed"""
    set_seed(seed)
    image_1 = do_inference(left_model, prompt=prompt, **kwargs).images
    set_seed(seed)
    image_2 = do_inference(right_model, prompt=prompt, **kwargs).images
    return (image_1, image_2)


@torch.no_grad()
def testing_model(
    pipeline_student: diffusers.StableDiffusionXLPipeline,
    pipeline_teacher: diffusers.StableDiffusionXLPipeline,
    prompts: Sequence[str],
    args: argparse.Namespace,
    nsamples: int = None,
    seed: int = 0,
    step: int = None,
):
    # just please don't ask
    vae_student = pipeline_student.vae
    pipeline_student.vae = pipeline_teacher.vae
    # monkey patching over
    # make with batches and potentially parralel
    metric_dict = defaultdict(list)
    perceptual_loss = lpips.LPIPS(net="vgg")

    logger.info("~" * 180)
    logger.info(f"Evaluating {len(prompts[:nsamples])} image pairs")
    for index in trange(0, len(prompts[:nsamples]), desc="calc metrics", leave=False):
        image1, image2 = generate_image_pairs(
            prompts[index],
            seed + index,
            pipeline_student,
            pipeline_teacher,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        image1_tensor = lpips.im2tensor(np.array(image1[0], dtype=np.uint8))
        image2_tensor = lpips.im2tensor(np.array(image2[0], dtype=np.uint8))
        metric_dict["MSE"].append(((image1_tensor - image2_tensor) ** 2).mean().item())
        metric_dict["MAE"].append((abs(image1_tensor - image2_tensor)).mean().item())
        metric_dict["LPIPS"].append(perceptual_loss.forward(image1_tensor, image2_tensor).item())

    for metric_name, metric_values in sorted(metric_dict.items()):
        logger.info(f"Mean {metric_name + ':':<6}  {np.mean(metric_values):.10f}")
        if args.wandb:
            wandb.log({f"Mean {metric_name}": np.mean(metric_values)}, step=step)
    ### Generate and save image pairs; evaluate (naive) automatic metrics

    grid = draw_grid(
        pipeline_student=pipeline_student,
        pipeline_teacher=pipeline_teacher,
        prompts=prompts,
        args=args,
        seed=seed,
        resize=512,
        num_images=args.visualize_nsamples,
    )
    if args.wandb:
        images = wandb.Image(grid, caption="Left: Student, Right: Teacher")
        wandb.log({"examples": images}, step=step)
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_path = os.path.join(args.save, f"image_pairs_step_{step}.png")
        grid.save(save_path)
        logger.info(f"Images from step {step} saved to {save_path}")

    pipeline_student.vae = vae_student


def get_linear_and_conv_order(
    pipeline: diffusers.StableDiffusionXLPipeline,
    args: argparse.Namespace,
):
    # layer filtering function
    def layer_filter_fn(layer: nn.Module, layer_name: str) -> bool:
        if isinstance(layer, (nn.Conv2d)):
            if min(layer.in_channels, layer.out_channels) < args.min_channels:
                return
        elif isinstance(layer, nn.Linear):
            if min(layer.in_features, layer.out_features) < args.min_channels:
                return
        else:
            return

        if args.layer_filter == "default":
            return re.search(DEFAULT_LAYER_REGEX, layer_name)
        if args.layer_filter == "all":
            return re.search(ALL_LAYER_REGEX, layer_name)
        if args.layer_filter == "linear_only":
            return re.search(LINEAR_LAYER_ONLY_REGEX, layer_name)
        if args.layer_filter == "conv_only":
            return re.search(CONV_LAYER_ONLY_REGEX, layer_name)
        else:
            # use custom layer filter
            return re.search(args.layer_filter, layer_name)

    # collect groups
    groups = []
    # collect from down blocks
    for i, block in enumerate(pipeline.unet.down_blocks):
        group = set()
        for module_name, module in block.named_modules():
            full_module_name = f"down_blocks.{i}.{module_name}"
            if layer_filter_fn(module, full_module_name):
                group.add((full_module_name, module))
        groups.append(group)
    # collect from mid block
    group = set()
    block = pipeline.unet.mid_block
    for module_name, module in block.named_modules():
        full_module_name = f"mid_block.{module_name}"
        if layer_filter_fn(module, full_module_name):
            group.add((full_module_name, module))
    groups.append(group)
    # collect from down blocks
    for i, block in enumerate(pipeline.unet.up_blocks):
        group = set()
        for module_name, module in block.named_modules():
            full_module_name = f"up_blocks.{i}.{module_name}"
            if layer_filter_fn(module, full_module_name):
                group.add((full_module_name, module))
        groups.append(group)
    return groups


def convert_to_dict(tuple_list):
    # Create a dictionary using the dict() constructor and a list comprehension
    dictionary = dict((key, value) for key, value in tuple_list)

    # Return the completed dictionary
    return dictionary


@torch.no_grad()
def draw_grid(
    pipeline_student: diffusers.StableDiffusionXLPipeline,
    pipeline_teacher: diffusers.StableDiffusionXLPipeline,
    prompts: Sequence[str],
    args: argparse.Namespace,
    seed: int = 0,
    resize: int = None,
    num_images: int = 8,
):
    vae_student = pipeline_student.vae
    pipeline_student.vae = pipeline_teacher.vae
    #### saving sample images
    images_1, images_2 = [], []
    for index in trange(0, len(prompts[:num_images]), desc="calc metrics", leave=False):
        (image1,), (image2,) = generate_image_pairs(
            prompts[index : index + 1],
            seed + index,
            pipeline_student,
            pipeline_teacher,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        images_1.append(image1)
        images_2.append(image2)
    pipeline_student.vae = vae_student
    return make_image_grid(
        list(chain.from_iterable(zip(images_1, images_2))), rows=len(images_1), cols=2, resize=resize
    )


@torch.no_grad()
def get_datasets_for_finetuning(
    pipeline: diffusers.StableDiffusionXLPipeline,
    prompts: Sequence[str],
    args: argparse.Namespace,
):
    ### getting datasets
    prompts_len = len(prompts)
    len_data_device = (prompts_len - 1) // len(args.devices) + 1
    datasets = []
    for i in range(len(args.devices)):
        start = i * len_data_device
        end = (i + 1) * len_data_device
        datasets.append(
            gather_unet_latents(
                pipeline,
                prompts[start:end],
                batch_size=args.finetune_batch_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                device="cpu",
            )
        )
    return datasets


def do_finetuning(
    pipeline_student: diffusers.StableDiffusionXLPipeline,
    pipeline_teacher: diffusers.StableDiffusionXLPipeline,
    prompts: Sequence[str],
    datasets: Any,
    args: argparse.Namespace,
):
    # this is intentional before every fine-tuning
    for _, param in pipeline_teacher.unet.named_parameters():
        param.requires_grad = False
    if args.finetune_method == "student":
        # monkey patching vae
        vae_student = pipeline_student.vae
        pipeline_student.vae = pipeline_teacher.vae
        datasets = get_datasets_for_finetuning(pipeline_student, prompts, args)
        pipeline_student.vae = vae_student

    assert datasets, "Empty datasets for fine-tuning"

    student_unet = pipeline_student.unet.float()

    student_unet = finetune_unet(
        student_unet=student_unet,
        teacher_unet=pipeline_teacher.unet,
        datasets=datasets,
        args=args,
        autocast_dtype=torch.bfloat16,
        scheduler=pipeline_teacher.scheduler,
        do_classifier_free_guidance=pipeline_teacher.do_classifier_free_guidance,
    )
    return student_unet.to(args.dtype)


def remove_snapshots(args):
    """Removing previous snapshots"""
    for path in [f for f in glob.glob(args.save + "/*.pickle")]:
        if os.path.isfile(path):
            os.remove(path)


def get_pipeline(scheduler_name: str, model_path: str, dtype: torch.dtype, device: str = "cuda"):
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
        model_path,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    if scheduler_name.lower() == "default":
        pass
    elif scheduler_name.lower() == "ddpm":
        pipe.scheduler = diffusers.DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif scheduler_name.lower() == "ddim":
        pipe.scheduler = diffusers.DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif scheduler_name.lower() == "heun":
        pipe.scheduler = diffusers.HeunDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif scheduler_name.lower() == "dpmsolver":
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif scheduler_name.lower() == "ode":
        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        pipe.scheduler.use_karras_sigmas = False
    elif scheduler_name is None:
        raise ValueError(f"{scheduler_name} is not specified. For default value use `--scheduler=defailt`")
    else:
        raise NotImplementedError(f"Unsupported scheduler_name value {scheduler_name}.")

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in AutoPipelineForText2Image.from_pretrained()",
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
        "--calibration_nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
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
        "--xtx_batch_size",
        type=int,
        default=4,
        help="do not quantize layers which have less than (this) input or output dimension",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="calibrate on timesteps less than this; based on pipeline.scheduler.config['num_train_timesteps']",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=0,
        help="calibrate on timesteps more or equal to this; based on pipeline.scheduler.config['num_train_timesteps']",
    )
    parser.add_argument(
        "--layer_filter",
        type=str,
        default="default",
        help="Regular expression that select layerst to quantize. "
        "By default all linear and conv layers in transformers and resnet block are quantized.",
    )
    parser.add_argument(
        "--min_channels",
        type=int,
        default=16,
        help="do not quantize layers which have less than (this) input or output dimension",
    )
    parser.add_argument(
        "--num_intermediate_finetunes",  # rename num_intermediate_finetunes - total number of times model is fine-tuned, including at the end. Internal behavior: linspace with (num + 1) points from 0 to num_subsets; skip first point. This ensures that the final round will take place after the entire model is quantized
        type=int,
        default=None,
        help="Total number of times model is fine-tuned, including at the end. Internal behavior: "
        "linspace with (num + 1) points from 0 to num_subsets; skip first point. "
        "This ensures that the final round will take place after the entire model is quantized",
    )
    parser.add_argument(
        "--snapshot_step",  # rename num_intermediate_finetunes - total number of times model is fine-tuned, including at the end. Internal behavior: linspace with (num + 1) points from 0 to num_subsets; skip first point. This ensures that the final round will take place after the entire model is quantized
        type=int,
        default=5,
        help="Do snapshot every snapshot_step",
    )
    parser.add_argument(
        "--eval_step",  # rename num_intermediate_finetunes - total number of times model is fine-tuned, including at the end. Internal behavior: linspace with (num + 1) points from 0 to num_subsets; skip first point. This ensures that the final round will take place after the entire model is quantized
        type=int,
        default=20,
        help="Do eval every eval_step",
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
    parser.add_argument(
        "--groupwise_codebooks",
        action="store_true",
        help="Whether to use codebook set for whole weights or for each input dim.Default use False.",
    )
    parser.add_argument(
        "--input_channel_scales",
        action="store_true",
        help="If True, use an additiona set of scales for input dimensions",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument("--loglevel", default="INFO", help="Logging level from ['WARN', 'INFO'(default), 'DEBUG']")

    parser = add_aqlm_engine_args(parser)
    args = parser.parse_args()
    args.code_dtype = getattr(torch, args.code_dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    loglevel = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.getLogger().setLevel(loglevel)
    logging.basicConfig(stream=sys.stdout, level=loglevel, format="[%(asctime)s][%(name)s] %(message)s")

    logger.info("=" * 180)
    logger.info(f"STARTING QUANTIZATION OF MODEL {args.model_path}")
    logger.info(f"Script arguments: {args}")

    if args.num_intermediate_finetunes == 0:
        args.num_intermediate_finetunes = None

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
            + f"_num_codebooks_{args.num_codebooks}"
            + f"_out_group_size_{args.out_group_size}"
            + f"_in_group_size_{args.in_group_size}"
            + f"_nbits_per_codebook_{args.nbits_per_codebook}"
            + f"_codebook_value_nbits_{args.codebook_value_nbits}"
            + f"_codebook_value_num_groups_{args.codebook_value_num_groups}"
            + f"_scale_nbits_{args.scale_nbits}"
            + f"_steps_per_epoch_{args.steps_per_epoch}"
            + f"_init_max_iter{args.init_max_iter}"
            + f"_{len(args.devices)}gpus"
        )
        args.group_size = args.in_group_size * args.out_group_size

        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
            settings=wandb.Settings(code_dir="."),
            save_code=True,
        )
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info("Generation seed:", seed)

    assert args.load is None, "TODO implement model loading"
    calibration_prompts = get_diffusion_prompts(
        args.calibration_prompts, nsamples=args.calibration_nsamples, seed=seed, eval_mode=False
    )
    evaluation_prompts = get_diffusion_prompts(
        args.evaluation_prompts, nsamples=args.evaluation_nsamples, seed=seed, eval_mode=True
    )
    if args.num_intermediate_finetunes:
        finetune_prompts = get_diffusion_prompts(
            args.calibration_prompts, nsamples=args.finetune_nsamples, seed=seed + 42, eval_mode=False
        )

    ### Load teacher and student models ###
    pipeline_teacher = get_pipeline(
        model_path=args.model_path, scheduler_name=args.scheduler, dtype=args.dtype, device=device
    )
    logger.info("Teacher pipeline loaded")
    pipeline_student = get_pipeline(
        model_path=args.model_path, scheduler_name=args.scheduler, dtype=args.dtype, device=device
    )
    start_step = 0
    if args.resume and args.save:
        os.makedirs(args.save, exist_ok=True)
        pickle_files_paths = [f for f in glob.glob(args.save + "/*.pickle")]
        assert (
            len(pickle_files_paths) < 2
        ), "Snapshoting does not support multiple files"  # make sure there isn't multiple pickle files
        if pickle_files_paths:
            pipeline_student.unet = torch.load(pickle_files_paths[0], map_location="cuda").to("cuda")
            start_step = int(pickle_files_paths[0].split("_")[-1].split(".")[0])

    if start_step != 0:
        logger.info(f"Continue quantizing from {start_step=}")
    else:
        logger.info("Quantizing from start")

    logger.info("Student pipeline loaded")

    if args.max_timestep is None:
        args.max_timestep = pipeline_teacher.scheduler.config["num_train_timesteps"]
    assert (
        args.max_timestep >= 100
    ), f"note that timesteps are {pipeline_teacher.scheduler.config['num_train_timesteps']}-based"

    ### Quantize model layers ###
    layer_order = get_linear_and_conv_order(pipeline_student, args)

    if args.wandb:
        df = wandb.Table(columns=["layer name"])

    datasets = []
    finetune_epochs = []
    if args.num_intermediate_finetunes:
        assert args.finetune_method in ["teacher", "student"], "Not implemented"
        assert args.num_intermediate_finetunes <= len(layer_order)

        finetune_epochs = np.linspace(
            0, len(layer_order) - 1, num=args.num_intermediate_finetunes + 1, dtype=int, endpoint=True
        )[1:]
        if args.finetune_method == "teacher":
            datasets = get_datasets_for_finetuning(pipeline_teacher, calibration_prompts, args)

    for step, list_of_layers in tqdm(enumerate(layer_order), desc="quantizing layers' subsets"):
        if step < start_step:
            continue

        subset = convert_to_dict(list_of_layers)

        if args.wandb:
            for name in subset.keys():
                df.add_data(name)
            wandb.log({"Quantization step": step}, commit=True)  # for correct wandb logging

        logger.info("=" * 180)
        logger.info(f"Quantizing subset {step} of {len(layer_order)}.")
        for layer_name, layer in subset.items():
            logger.info(f"{layer_name}: {layer}")
        subset_start_time = time.perf_counter()

        quantize_unet_layers_inplace_(
            pipeline_student,
            calibration_prompts,
            subset,
            teacher_unet=pipeline_teacher.unet,
            args=args,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            subset_step=step,
        )  # updates pipeline_student in-place
        logger.info(f"Subset {step} of {len(layer_order)} quantized in {time.perf_counter() - subset_start_time:.1f}s.")

        # Finetuning step
        if step in finetune_epochs:
            # this is intentional before every finetuning
            logger.info(f"Doing finetuning at subset number {step}")
            pipeline_student.unet = do_finetuning(pipeline_student, pipeline_teacher, finetune_prompts, datasets, args)

        # Evaluation
        if step % args.snapshot_step == 0 and args.resume and args.save:
            logger.info(f"Saving snapshot_{step}")
            os.makedirs(args.save, exist_ok=True)
            remove_snapshots(args)
            torch.save(pipeline_student.unet, os.path.join(args.save, f"quantized_unet_{step+1}.pickle"))
            if args.on_save:
                exec(args.on_save)  # a callback e.g. to save progress in slurm or similar distributed infrastructure

        if step % args.eval_step == 0:
            eval_start_time = time.perf_counter()
            testing_model(
                pipeline_student,
                pipeline_teacher,
                evaluation_prompts,
                args=args,
                nsamples=args.evaluation_nsamples,
                seed=seed,
                step=step,
            )
            elapsed = time.perf_counter() - eval_start_time
            logger.info(f"Evaluation time after step {step}: {elapsed:.1f}s.")

    if args.wandb:
        wandb.log({"List order": df})

    # monkey patching starting
    # just please don't ask
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
        logger.info("Images saved to" + os.path.join(args.save, "image_pairs.png"))

    # monkey patching over
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        remove_snapshots(args)
        torch.save(pipeline_student.unet, os.path.join(args.save, "quantized_unet.pickle"))
        if args.on_save:
            exec(args.on_save)
