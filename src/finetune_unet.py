from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import diffusers.schedulers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import Gather

from aq_engine import replace_parameter_
from latents_dataset import DiffusionLatentsDataset
from src.nested import nested_map


@torch.enable_grad()
def finetune_unet(
    *,
    student_unet: nn.Module,
    teacher_unet: nn.Module,
    datasets: Sequence[DiffusionLatentsDataset],
    args: Namespace,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Fine-tune a module with pre-quantized linear layers so as to minimize MSE between layer-wise inps/outs

    :param student_unet: a trainable module where linear layers are replaced by QuantizedLinear instances
    :param datasets: one or several datasets of unet inputs; the must always be one dataset per each device, acting as shards
    :param args: quantization hyperparameters from main.py
    :param kwargs: additional keyword arguments to be passed into layer on each forward
    """
    assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
    assert isinstance(datasets, (list, tuple)) and len(datasets) == len(args.devices)
    for i in range(len(args.devices)):
        assert abs(len(datasets[i]) - len(datasets[0])) <= datasets[0].num_inference_steps

    teacher_and_student = nn.ModuleDict(dict(teacher=teacher_unet, student=student_unet))

    replicas = kwargs_by_device = None
    if len(args.devices) > 1:
        replicas = torch.nn.parallel.replicate(teacher_and_student, args.devices)
        replicas[0] = teacher_and_student

        kwargs_by_device = []
        for device in args.devices:
            kwargs_by_device.append(
                {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            )

    for name, param in teacher_unet.named_parameters():
        assert not param.requires_grad, f"Teacher parameter {name} requires_grad, but it shouldn't"

    # initialize trainable parameters on main device; prepare to send them to replicas
    trainable_parameters_by_name = {
        name: param for name, param in teacher_and_student.named_parameters() if param.requires_grad
    }
    param_names, trainable_parameters = zip(*trainable_parameters_by_name.items())
    trainable_parameters = nn.ParameterList(trainable_parameters)
    for param in trainable_parameters:
        param.grad = torch.zeros_like(param)
    if replicas:
        replacement_tables = _make_parameter_replacement_tables(
            teacher_and_student, replicas, param_names, trainable_parameters
        )

    print(f"Fine-tuning {sum(param.numel() for param in trainable_parameters)} parameters")
    opt = torch.optim.Adam(
        trainable_parameters, lr=args.finetune_lr, betas=(args.finetune_adam_beta1, args.finetune_adam_beta2)
    )

    # backup best parameters
    if args.finetune_keep_best:
        best_parameters = deepcopy(trainable_parameters)

    assert args.finetune_batch_size % len(args.devices) == 0, "batch_size must be divisible by the number of GPUs"

    num_samples_per_device = min(map(len, datasets))
    local_batch_size = args.local_batch_size
    if local_batch_size is None:
        local_batch_size = args.finetune_batch_size // len(args.devices)

    assert (
        args.finetune_batch_size % (local_batch_size * len(args.devices)) == 0
    ), "full batch size must be divisible by number of gpus times local batch size "
    num_accumulation_steps = args.finetune_batch_size // (local_batch_size * len(args.devices))
    assert num_samples_per_device % local_batch_size * num_accumulation_steps == 0, (
        num_samples_per_device,
        local_batch_size,
    )
    steps_per_epoch = num_samples_per_device // local_batch_size

    batch_iterators = [
        datasets[i].iterate_minibatches(batch_size=local_batch_size, device=args.devices[i], allow_incomplete=True)
        for i in range(len(args.devices))
    ]

    previous_best_loss = float("inf")  # for early stopping
    steps_accumulated = 0
    for epoch in range(args.finetune_max_epochs):
        loss_numerator = loss_denominator = 0
        for step in range(steps_per_epoch):
            if len(args.devices) == 1:
                loss = _compute_mse_on_batch(teacher_and_student, batch_iterators[0], **kwargs)
            else:
                loss = _compute_mse_parallel(
                    args.devices,
                    replicas,
                    trainable_parameters,
                    replacement_tables,
                    batch_iterators,
                    kwargs_by_device,
                )

            (loss / num_accumulation_steps).backward()
            steps_accumulated += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")
            if steps_accumulated >= num_accumulation_steps:
                opt.step()
                opt.zero_grad()
                steps_accumulated = 0

            loss_numerator += loss.item()
            loss_denominator += 1
            if verbose and (epoch * steps_per_epoch + step) % args.print_frequency == 0:
                print(f"epoch={epoch}\tstep={step}\tloss={loss_numerator / loss_denominator:.10f}\t")

        if verbose and (epoch * steps_per_epoch + step) % args.print_frequency != 0:
            print(f"epoch={epoch}\tstep={step}\tloss={loss_numerator / loss_denominator:.10f}\t")

        if args.finetune_relative_mse_tolerance is not None:
            epoch_loss = loss_numerator / loss_denominator
            if args.finetune_keep_best:
                if epoch_loss / previous_best_loss < 1.0:
                    best_parameters = deepcopy(trainable_parameters)
                else:
                    trainable_parameters = best_parameters
            if epoch_loss / previous_best_loss > (1.0 - args.finetune_relative_mse_tolerance):
                return student_unet  # early stopping; no updates after last epoch's beam search
            previous_best_loss = min(epoch_loss, previous_best_loss)
    opt.zero_grad(set_to_none=True)
    return student_unet


def _make_parameter_replacement_tables(
    layer: nn.Module, replicas: Sequence[nn.Module], param_names: Sequence[str], parameters: nn.ParameterList
) -> Sequence[List[Sequence[Tuple[nn.Module, str]]]]:
    """
    Prepare auxiliary data structures for quickly copying parameters to replicas for data-parallel training.

    """
    assert len(param_names) == len(parameters)
    assert len(replicas) > 1
    assert replicas[0] is layer

    parameters_by_name = dict(zip(param_names, parameters))

    param_to_name = {param: name for name, param in parameters_by_name.items()}
    param_occurences = defaultdict(list)  # param_name -> List [ Tuple [submodule name, attr name] ]
    for submodule_name, submodule in layer.named_modules():
        for attr_name, param in submodule.named_parameters(recurse=False):  # immediate params (excluding children)
            if param in param_to_name:
                param_name = param_to_name[param]
                param_occurences[param_name].append((submodule_name, attr_name))
    assert len(param_occurences) == len(parameters), "internal error: not all parameters were found"

    replacement_tables = []
    for replica in replicas:
        replacement_table = list()  # for each master param -> List[ Tuple[replica submodule, attr name] ]
        replica_modules_by_name: Dict[str, nn.Module] = dict(replica.named_modules())

        for param_name, master_param in zip(param_names, parameters):
            param_replacements = list()
            for submodule_name, attr_name in param_occurences[param_name]:
                param_replacements.append((replica_modules_by_name[submodule_name], attr_name))
            replacement_table.append(param_replacements)
        replacement_tables.append(replacement_table)
    return replacement_tables


def _compute_mse_on_batch(
    teacher_and_student: nn.Module,
    batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    *,
    do_classifier_free_guidance: bool,
    scheduler: diffusers.schedulers.SchedulerMixin,
    teacher_dtype: Optional[torch.dtype] = None,
    student_dtype: Optional[torch.dtype] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Compute the activation MSE error between transformer layers
    :param
    """
    t, latents, unet_kwargs = next(batch_iter)
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    with torch.cuda.amp.autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype, cache_enabled=True):
        with torch.no_grad():
            if autocast_dtype is None and teacher_dtype is not None:
                latent_model_input, t, unet_kwargs = nested_map(
                    lambda x: x.to(teacher_dtype) if isinstance(x, torch.Tensor) else x,
                    (latent_model_input, t, unet_kwargs),
                )

            target = teacher_and_student["teacher"](latent_model_input, t, **unet_kwargs, return_dict=False)[0]

        if autocast_dtype is None and teacher_dtype is not None:
            latent_model_input, t, unet_kwargs = nested_map(
                lambda x: x.to(student_dtype) if isinstance(x, torch.Tensor) else x,
                (latent_model_input, t, unet_kwargs),
            )
        pred = teacher_and_student["student"](latent_model_input, t, **unet_kwargs, return_dict=False)[0]
    return F.mse_loss(pred, target.to(pred.dtype))


def _compute_mse_parallel(
    devices: Sequence[torch.device],
    replicas: Sequence[nn.Module],
    parameters_to_replicate: nn.ParameterList,
    replacement_tables: Sequence[List[Sequence[Tuple[nn.Module, str]]]],
    batch_iterators: Sequence[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
    kwargs_by_device: Sequence[Dict[str, Any]],
) -> torch.Tensor:
    """Compute MSE in parallel over multiple GPUs, each GPU processes a portion of samples"""
    replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
    funcs_by_replica = [_compute_mse_on_batch for _ in replicas]
    inputs_by_replica = []
    for i in range(len(devices)):
        if i != 0:  # no overrides needed for master module
            for replacement_param, replacement_table in zip(replicated_parameters[i], replacement_tables[i]):
                for (replica_submodule, attr_name) in replacement_table:
                    replace_parameter_(replica_submodule, attr_name, replacement_param)
        inputs_by_replica.append((replicas[i], batch_iterators[i]))
    mse_components = torch.nn.parallel.parallel_apply(
        funcs_by_replica, inputs_by_replica, kwargs_by_device, devices=devices
    )
    return Gather.apply(devices[0], 0, *(mse.view(1) for mse in mse_components)).mean()
