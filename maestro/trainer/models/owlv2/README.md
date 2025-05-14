# OWLv2 Model Integration for Maestro

This module provides support for fine-tuning and inference with the OWLv2 model for open vocabulary object detection, following the same structure and conventions as other Maestro models (e.g., Florence-2).

## Files
- `core.py`: Model configuration, dataset class, and training logic
- `entrypoint.py`: CLI entrypoint for training and inference
- `inference.py`: Inference utilities for running predictions
- `loaders.py`: Data loading utilities, including support for N-channel and high bit-depth TIFFs (where possible)
- `__init__.py`: Module init

## Features
- Fine-tuning on custom datasets (JSONL referencing 3-channel images)
- (Planned/Partial) Support for N-channel and high bit-depth TIFFs
- CLI for training and inference
- Example notebook and tests

## Usage

### Training
To train OWLv2 on your dataset (JSONL referencing images):

```bash
python -m maestro.trainer.models.owlv2.entrypoint train-model --jsonl-path /path/to/annotations.jsonl --images-dir /path/to/images
```

- `--jsonl-path`: Path to your JSONL annotation file
- `--images-dir`: Directory containing your images
- `--model-name`: (optional) Hugging Face model name (default: google/owlv2-base-patch16)
- `--device`: (optional) cuda or cpu
- `--learning-rate`, `--epochs`, `--batch-size`, `--output-dir`: (optional) Training hyperparameters

### Inference
To run inference on a single image:

```bash
python -m maestro.trainer.models.owlv2.entrypoint infer --model-path ./owlv2_output --processor-path ./owlv2_output --image-path /path/to/image.jpg --prompts "cat,dog,car"
```

- `--model-path`: Path to the trained model directory
- `--processor-path`: Path to the processor directory (usually same as model)
- `--image-path`: Path to the image file
- `--prompts`: Comma-separated list of text prompts (categories)
- `--device`: (optional) cuda or cpu

## Notes
- 3-channel images (e.g., RGB) are supported. N-channel and high bit-depth TIFFs may require additional dependencies or preprocessing. See `loaders.py` for details and limitations.
- The logic and workflow are closely aligned with the Florence-2 model for consistency.

## Example
See `cookbooks/maestro_owlv2_object_detection.ipynb` for a full example of fine-tuning and inference.
