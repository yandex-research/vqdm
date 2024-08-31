from typing import List, Tuple, Union

import diffusers
import torch
import torch.nn.functional as F

from decomposed_inference import do_inference
from src.nested import nested_compare, nested_flatten, nested_map
from src.utils import iterate_minibatches


class DiffusionLatentsDataset(torch.utils.data.Dataset):
    def __init__(self, timesteps, latents, **kwargs):
        """A dataset with latent vectors and kwargs for calling a diffusion unet"""
        super().__init__()
        self.timesteps, self.latents, self.kwargs = timesteps, latents, kwargs
        self.num_prompts, self.num_inference_steps, *_ = self.latents.shape

    def __len__(self) -> int:
        return self.num_prompts * self.num_inference_steps

    def __getitem__(self, index: Union[int, torch.IntTensor]) -> Tuple[torch.Tensor, ...]:
        prompt_index = index // self.num_inference_steps
        timestep_index = index % self.num_inference_steps
        return self._load_sample(prompt_index, timestep_index)

    def _load_sample(
        self, prompt_indices: torch.LongTensor, timestep_indices: torch.LongTensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Gather a batch of unet inputs for the corresponding prompt and scheduler step
        :returns: t, latents, unet_kwargs, to be used as:

        >>> t, latents, unet_kwargs = _load_sample(...)
        >>> latent_model_input = torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents
        >>> latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        >>> pred = pipeline.unet(latent_model_input, t, **unet_kwargs, return_dict=False)[0]
        >>> print("latent noise prediction for next step:", pred)

        """
        timesteps = self.timesteps[prompt_indices, timestep_indices]
        latents = self.latents[prompt_indices, timestep_indices]

        def _load_kwarg_for_indices(buffer):
            if isinstance(buffer, torch.Tensor):
                chosen = buffer[prompt_indices].swapaxes(0, 1)  # swap [batch] and [negative/positive] dims
                return chosen.flatten(0, 1)  # concat all negative prompts first, then all positive
            else:
                return buffer  # buffer is a non-tensor, e.g. None

        kwargs = nested_map(_load_kwarg_for_indices, self.kwargs)
        if timesteps.numel() == 1:
            timesteps = timesteps.reshape([])
        else:
            timesteps = torch.cat([timesteps, timesteps])  # repeat for negative and positive embeds
        return timesteps, latents, kwargs

    def iterate_minibatches(self, batch_size: int, device: torch.device, allow_incomplete: bool):
        """Iterate batches faster than DataLoader; Note: this function duplicates the code from utils.py:iterate_minibatches"""
        num_samples = len(self)
        indices = torch.randperm(num_samples, device=self.timesteps.device)
        while True:
            prev_batch = None
            for batch_start in range(0, len(indices), batch_size):
                if not allow_incomplete and batch_start + batch_size > len(indices):
                    break
                batch_ix = indices[batch_start : batch_start + batch_size]

                batch = nested_map(
                    lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, self[batch_ix]
                )

                if prev_batch is not None:
                    yield prev_batch
                prev_batch = batch
                del batch
            yield prev_batch


def gather_unet_latents(
    pipeline: diffusers.StableDiffusionXLPipeline,
    prompts: List[str],
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
) -> DiffusionLatentsDataset:
    all_timesteps, all_latents, all_kwargs = None, None, None

    for start in range(0, len(prompts), batch_size):
        batch_indices = slice(start, start + batch_size)
        (outputs,) = do_inference(
            pipeline,
            prompts[batch_indices],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="latent",
            return_intermediate_latents=True,
            return_dict=False,
        )

        assert len(outputs) == num_inference_steps
        for record_index, (t, latents, kwargs) in enumerate(outputs):
            assert t.numel() == 1
            assert latents.shape[0] == batch_size  # regardless of num_prompt_parts
            if all_timesteps is None:
                all_timesteps = torch.zeros(len(prompts), num_inference_steps, dtype=t.dtype, device=device)
                all_latents = torch.zeros(
                    len(prompts), num_inference_steps, *latents.shape[1:], dtype=latents.dtype, device=device
                )
                num_prompt_parts = (
                    2 if pipeline.do_classifier_free_guidance else 1
                )  #  either just the prompt or negative+positive prompt pair
                all_kwargs = nested_map(
                    lambda v: (  # create a save buffer for each tensor in kwargs
                        torch.zeros(len(prompts), num_prompt_parts, *v.shape[1:], dtype=v.dtype, device=device)
                        if isinstance(v, torch.Tensor)
                        else v
                    ),
                    kwargs,
                )

            all_timesteps[batch_indices, record_index] = t
            all_latents[batch_indices, record_index] = latents

            def _save_kwarg_inplace(buffer, new_value):
                if isinstance(new_value, torch.Tensor) and record_index == 0:
                    buffer[batch_indices] = new_value.reshape(2, batch_size, *new_value.shape[1:]).swapaxes(0, 1)

                elif isinstance(new_value, torch.Tensor) and record_index != 0:
                    # all indices should have the same kwargs; check this
                    assert torch.allclose(
                        buffer[batch_indices],
                        new_value.reshape(2, batch_size, *new_value.shape[1:]).swapaxes(0, 1).to(device),
                    )
                else:
                    assert buffer == new_value

            nested_map(_save_kwarg_inplace, all_kwargs, kwargs)
    return DiffusionLatentsDataset(all_timesteps, all_latents, **all_kwargs)


def concatenate_latent_datasets(*datasets: DiffusionLatentsDataset) -> DiffusionLatentsDataset:
    """Concatenate multiple latent datasets over the batch axis"""
    for dataset in datasets:
        if dataset.num_inference_steps != datasets[0].num_inference_steps:
            raise ValueError(
                f"Expected all datasets to have the same num_inference_steps "
                f"but found {datasets[0].num_inference_steps} and {dataset.num_inference_steps}"
            )
        assert nested_compare(dataset.kwargs, datasets[0].kwargs), "dataset kwargs have different structure"

        def _assert_values_are_compatible(v1, v2):
            if isinstance(v1, torch.Tensor):
                assert isinstance(v2, torch.Tensor)
                assert v1.shape[1:] == v2.shape[1:]
            else:
                assert not isinstance(v2, torch.Tensor)
                assert v1 == v2

        # v-- if values cannot be merged safely, this raises AssertionError
        nested_map(_assert_values_are_compatible, dataset.kwargs, datasets[0].kwargs)

    combined_timesteps = torch.cat([dataset.timesteps for dataset in datasets], dim=0)

    combined_latents = torch.cat([dataset.latents for dataset in datasets], dim=0)

    combined_kwargs = nested_map(
        lambda *values: torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values[0],
        *[dataset.kwargs for dataset in datasets],
    )
    return DiffusionLatentsDataset(combined_timesteps, combined_latents, **combined_kwargs)


def test_dataset_merging(pipeline, prompts, seed):
    torch.manual_seed(seed)
    dataset = gather_unet_latents(pipeline, prompts, batch_size=1, num_inference_steps=50, device="cpu")

    torch.manual_seed(seed)
    dataset_parts = []
    for i in range(len(prompts)):
        dataset_parts.append(
            gather_unet_latents(pipeline, [prompts[i]], batch_size=1, num_inference_steps=50, device="cpu")
        )

    concatenated_dataset = concatenate_latent_datasets(*dataset_parts)
    assert torch.allclose(concatenated_dataset.timesteps, dataset.timesteps, atol=1e-3).item()
    assert torch.allclose(concatenated_dataset.latents, dataset.latents, atol=1e-3).item()
    assert all(
        nested_flatten(
            nested_map(
                lambda v1, v2: torch.allclose(v1, v2, atol=1e-3).item() if isinstance(v1, torch.Tensor) else v1 == v2,
                concatenated_dataset.kwargs,
                dataset.kwargs,
            )
        )
    )


def test_training_on_latents(student_pipeline, teacher_pipeline, prompts, seed):
    torch.manual_seed(1337)
    dataset = gather_unet_latents(
        student_pipeline, prompts, batch_size=4, num_inference_steps=50, device=torch.device("cpu")
    )
    #                                 ^-- or teacher pipeline, depending on behavior cloning vs DAgger

    device = next(student_pipeline.unet.parameters()).device

    t, latents, unet_kwargs = dataset._load_sample([0, 1], [12, 34])
    t, latents = t.to(device), latents.to(device)
    unet_kwargs = nested_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, unet_kwargs)
    latent_model_input = torch.cat([latents] * 2) if student_pipeline.do_classifier_free_guidance else latents
    latent_model_input = student_pipeline.scheduler.scale_model_input(latent_model_input, t)

    pred = student_pipeline.unet(latent_model_input, t, **unet_kwargs, return_dict=False)[0]

    with torch.no_grad():
        target = teacher_pipeline.unet(latent_model_input, t, **unet_kwargs, return_dict=False)[0]

    loss = F.mse_loss(pred, target)
    loss.backward()
