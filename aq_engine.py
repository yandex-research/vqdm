from __future__ import annotations

import logging
import math
import random
import sys
import time
from argparse import Namespace
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import Gather

from src.aq import QuantizedWeight
from src.utils import ellipsis

logger = logging.getLogger("aqe")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="[%(asctime)s][aqe] %(message)s")


class AQEngine(nn.Module):
    """A wrapper class that runs AQ training for a single linear layer. All the important math is in aq.py"""

    def __init__(self, layer: Union[nn.Linear, nn.Conv2d], accumultor_dtype: torch.dtype = torch.float64):
        super().__init__()
        self.layer = layer
        self.device = self.layer.weight.device
        self.columns = self.reference_weight.shape[1]
        self.register_buffer(
            "XTX", torch.zeros((self.columns, self.columns), dtype=accumultor_dtype, device=self.device)
        )
        self.quantized_weight: Optional[QuantizedWeight] = None
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """Accumulate a minibatch of layer inputs and update the X.T @ X (aka half hessian)"""
        assert self.XTX is not None, "Already ran quantization; cannot add more data batches"
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        else:
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        new_samples = inp.shape[1]
        self.XTX *= self.nsamples / (self.nsamples + new_samples)
        self.nsamples += new_samples
        inp = math.sqrt(1 / self.nsamples) * inp.to(self.XTX.dtype)
        self.XTX += inp.matmul(inp.t())

    @property
    @torch.no_grad()
    def reference_weight(self):
        return self.layer.weight.detach().to(dtype=torch.float32).flatten(1)

    @torch.enable_grad()
    def quantize(self, *, args: Namespace, verbose: bool = True) -> QuantizedWeight:
        """create a QuantizedLinear with specified args based on the collected hessian (XTX) data"""
        assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
        assert args.devices[0] == self.device, (args.devices[0], self.XTX.device)

        if args.group_channels and isinstance(self.layer, torch.nn.Conv2d):
            with torch.no_grad():
                assert isinstance(self.layer, torch.nn.Conv2d)
                _, in_channels, height, width = self.layer.weight.shape
                # permute hessian to accomodate channels-last activations
                self.XTX = self.XTX.reshape((in_channels, height, width, in_channels, height, width))
                #                                 0         1       2       3           4      5
                self.XTX = self.XTX.permute(1, 2, 0, 4, 5, 3).reshape(
                    height * width * in_channels, height * width * in_channels
                )

                self.layer.weight = nn.Parameter(
                    self.layer.weight.permute(0, 2, 3, 1), requires_grad=self.layer.weight.requires_grad
                )
                # ^-- new shape: [out_channels, h, w, in_channels] - see self.reference_weight above

        self.quantized_weight = QuantizedWeight(
            XTX=self.XTX.to(device=self.device, dtype=torch.float32),
            reference_weight=self.reference_weight,
            out_group_size=args.out_group_size,
            in_group_size=args.in_group_size,
            num_codebooks=args.num_codebooks,
            nbits_per_codebook=args.nbits_per_codebook,
            groupwise_codebooks=args.groupwise_codebooks,
            codebook_value_nbits=args.codebook_value_nbits,
            codebook_value_num_groups=args.codebook_value_num_groups,
            scale_nbits=args.scale_nbits,
            input_channel_scales=args.input_channel_scales,
            max_iter=args.init_max_iter,
            max_points_per_centroid=args.init_max_points_per_centroid,
            code_dtype=args.code_dtype,
            devices=args.devices,
            verbose=True,
        )
        print(flush=True)

        differentiable_parameters = nn.ParameterDict(
            {name: param for name, param in self.quantized_weight.named_parameters() if param.requires_grad}
        )
        opt = torch.optim.Adam(differentiable_parameters.values(), lr=args.lr, betas=(0.0, 0.95), amsgrad=True)

        replicas = None
        if len(args.devices) > 1:
            replicas = torch.nn.parallel.replicate(self, args.devices)
            replicas[0] = self

        previous_best_loss = float("inf")  # for early stopping
        for epoch in range(args.max_epochs):
            # train codebooks and scales
            logger.info(f"Epoch {epoch}:")
            epoch_start_time = time.perf_counter()
            step_losses = []
            for step in range(args.steps_per_epoch):
                if len(args.devices) == 1:
                    loss = self._compute_mse()
                else:
                    loss = self._compute_mse_parallel(args.devices, replicas, differentiable_parameters)

                if not torch.isfinite(loss).item():
                    raise ValueError(f"Quantization loss is {loss}")
                if step == 0 and args.relative_mse_tolerance is not None:
                    if loss.item() / previous_best_loss > (1.0 - args.relative_mse_tolerance):
                        logger.info(f"Early stopping at {epoch=}")
                        return self.quantized_weight  # early stopping; no updates after last epoch's beam search
                    previous_best_loss = min(previous_best_loss, loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()
                step_losses.append(loss.item())

            # search for better codes (cluster indices)
            seed = random.getrandbits(256)

            elapsed = time.perf_counter() - epoch_start_time
            logger.info(
                " " * 4
                + f"steps={step+1} Losses: first={step_losses[0]:.10f} max={max(step_losses):.10f} min={min(step_losses):.10f} last={step_losses[-1]:.10f}  t={elapsed:.3f}s."
            )
            logger.debug(
                " " * 8
                + f"step losses: {' '.join([f'{step_losses[x]:.10f}' for x in range(0, len(step_losses), args.print_frequency)])}"
            )

            self.beam_search_update_codes_(
                args.devices,
                replicas,
                differentiable_parameters,
                seed=seed,
                beam_size=args.beam_size,
                verbose=True,
            )
        if args.group_channels and isinstance(self.layer, torch.nn.Conv2d):
            with torch.no_grad():
                assert isinstance(self.layer, torch.nn.Conv2d)
                # restore previous permutation
                self.XTX = self.XTX.reshape((height, width, in_channels, height, width, in_channels))
                #                              0      1       2           3      4          5
                self.XTX = self.XTX.permute(2, 0, 1, 5, 3, 4).reshape(
                    in_channels * height * width, in_channels * height * width
                )
                self.layer.weight = nn.Parameter(
                    self.layer.weight.permute(0, 3, 1, 2), requires_grad=self.layer.weight.requires_grad
                )
        return self.quantized_weight

    def _compute_mse(self, selection: Union[slice, ellipsis] = ...) -> torch.Tensor:
        """
        Compute the activation MSE error = ||X @ quantized_weight - X @ reference_weight||^2
        Use the square-of-difference formula to avoid materializing per-batch predictions
        :param selection:  By default, compute MSE normally. If selection is specified, this method will instead
            compute MSE over a portion of output channels that align with the selected out_groups (for parallelism)
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )
        """
        assert self.quantized_weight is not None, "must be called inside / after AQUtil.quantize"
        quantized_weight = self.quantized_weight(selection)

        if isinstance(selection, ellipsis):
            reference_weight = self.reference_weight.detach().to(quantized_weight.dtype)
        else:
            assert isinstance(selection, slice)
            out_channel_selection = slice(
                selection.start * self.quantized_weight.out_group_size,
                selection.stop * self.quantized_weight.out_group_size,
            )

            reference_weight = self.reference_weight.detach()[out_channel_selection].to(quantized_weight.dtype)
        delta_weight = (quantized_weight - reference_weight).to(self.XTX.dtype)
        return (delta_weight @ self.XTX).flatten() @ delta_weight.flatten() / self.quantized_weight.out_features

    def _replace_and_compute_mse(self, params_to_replace: nn.ParameterDict, selection: slice) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then compute MSE"""
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        return self._compute_mse(selection)

    def _compute_mse_parallel(
        self, devices: Sequence[torch.device], replicas: Sequence[AQEngine], parameters_to_replicate: nn.ParameterDict
    ) -> torch.Tensor:
        """Compute MSE in parallel over output channels"""
        replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
        num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
        shard_size = (num_output_groups - 1) // len(devices) + 1
        active_slices_by_replica = [
            slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
        ]
        funcs_by_replica = [replica._replace_and_compute_mse for replica in replicas]
        inputs_by_replica = [(dict(), active_slices_by_replica[0])]  # no replacements needed for 0-th replica (master)
        for i in range(1, len(devices)):
            inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
        mse_components = torch.nn.parallel.parallel_apply(funcs_by_replica, inputs_by_replica, devices=devices)
        return Gather.apply(devices[0], 0, *(mse.view(1) for mse in mse_components)).sum()

    def _replace_and_beam_search(self, params_to_replace: nn.ParameterDict, selection: slice, **kwargs) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then run beam search"""
        dtype = self.quantized_weight.codebooks.dtype
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        out_channel_selection = slice(
            selection.start * self.quantized_weight.out_group_size,
            selection.stop * self.quantized_weight.out_group_size,
        )
        reference_weight = self.reference_weight.detach()[out_channel_selection].to(dtype)
        return self.quantized_weight.beam_search_update_codes_(
            self.XTX.to(dtype), reference_weight, selection=selection, **kwargs
        ).clone()

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        devices: Sequence[torch.device],
        replicas: Sequence[AQEngine],
        parameters_to_replicate: nn.ParameterDict,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Update self.quantized_weight codes in-place via beam search"""
        if len(devices) == 1:  # single device
            assert replicas is None
            dtype = self.quantized_weight.codebooks.dtype
            self.quantized_weight.beam_search_update_codes_(
                self.XTX.to(dtype), self.reference_weight.detach().to(dtype), dim_rng=random.Random(seed), **kwargs
            )
        else:
            assert replicas[0] is self
            replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices)
            num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
            shard_size = (num_output_groups - 1) // len(devices) + 1
            active_slices_by_replica = [
                slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
            ]

            funcs_by_replica = [replica._replace_and_beam_search for replica in replicas]
            inputs_by_replica = [(dict(), active_slices_by_replica[0])]
            for i in range(1, len(devices)):
                inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
            kwargs_by_replica = [dict(kwargs, dim_rng=random.Random(seed)) for _ in range(len(devices))]
            new_code_parts_by_replica = torch.nn.parallel.parallel_apply(
                funcs_by_replica, inputs_by_replica, kwargs_by_replica, devices=devices
            )
            # gather all code parts and assign them to each replica
            for device, replica in zip(devices, replicas):
                replica.quantized_weight.set_codes(Gather.apply(device, 0, *new_code_parts_by_replica))


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    """A hacky way to substitute an already registered parameter with a non-parameter tensor. Breaks future use."""
    if name in module._parameters:
        module._parameters[name] = new_value
    else:
        setattr(module, name, new_value)
