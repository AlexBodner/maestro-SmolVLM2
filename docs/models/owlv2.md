# OWLv2 Object Detection Model

This page will document how to use the OWLv2 model for open vocabulary object detection and fine-tuning in Maestro.

## Features
- Fine-tuning on custom datasets (JSONL referencing 3-channel images)
- (Planned) Support for N-channel and high bit-depth TIFFs

## Usage

Instructions coming soon.

## How to Use

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

**Note:**
- 3-channel images (e.g., RGB) are supported. N-channel and high bit-depth TIFFs may require additional dependencies or preprocessing.
- See the code and loaders for current limitations.

## Planned Features
- N-channel and high bit-depth TIFF support (if possible)
- Example notebook and tests
