<a href='https://arxiv.org/abs/2409.00492'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
<a href='https://yandex-research.github.io/vqdm/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 

Official repository with code from [VQDM: Accurate Compression of Text-to-Image Diffusion Models via Vector Quantization](https://arxiv.org/abs/2409.00492) paper.

## Usage

### Dependencies

Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```
### How to quantize a model with VQDM
Please see examples of quantization, finetuning, metric calculation and image_generation scripts in the repository.
For quantization - `quantization.sh`
For finetuning - `finetuning_quantization.sh`
For image generation on a set of prompts - `eval_image_generation.sh`
For metric calculation - `metric_calculation.sh`

```
Main CLI arguments:
- `CUDA_VISIBLE_DEVICES` - by default, the code will use all available GPUs. If you want to use specific GPUs (or one GPU), set up this variable.
- `--calibration/finetune/evaluation nsamples` - the number of calibration/finetune/evaluation prompts
- `--num_codebooks` - number of codebooks per layer
- `--nbits_per_codebook` - each codebook will contain 2 ** nbits_per_codebook vectors
- `--in_group_size` - how many weights are quantized together (aka "g" in the arXiv paper)
- `--finetune_batch_size` - (for fine-tuning only) the total number of sequences used for each optimization step
- `--local_batch_size` - when accumulating finetune_batch_size, process this many samples per GPU per forward pass (affects GPU RAM usage)
- `--relative_mse_tolerance`- (for initial calibration) - stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)
- `--save` -- path to save/load quantized model. (see also: `--load`)
- `--wandb` - if this parameter is set, the code will log results to wandb

There are additional hyperparameters aviailable. Run `python main.py and main_text2image.py  --help` for more details on command line arguments.
```

The  code is based on  [Extreme Compression of Large Language Models via Additive Quantization](https://github.com/Vahe1994/AQLM) 

### BibTeX
```
@misc{egiazarian2024accuratecompressiontexttoimagediffusion,
      title={Accurate Compression of Text-to-Image Diffusion Models via Vector Quantization}, 
      author={Vage Egiazarian and Denis Kuznedelev and Anton Voronov and Ruslan Svirschevski and Michael Goin and Daniil Pavlov and Dan Alistarh and Dmitry Baranchuk},
      year={2024},
      eprint={2409.00492},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.00492}, 
}
```
