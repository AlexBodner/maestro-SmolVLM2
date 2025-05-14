# Placeholder for OWLv2 CLI entrypoint (training, evaluation)

import typer
from .core import OWLv2Configuration, train, OWLv2JSONLDataset
from .inference import run_inference

owlv2_app = typer.Typer(help="Fine-tune and evaluate OWLv2 model")

@owlv2_app.command()
def train_model(
    jsonl_path: str,
    images_dir: str,
    model_name: str = "google/owlv2-base-patch16",
    device: str = "cuda",
    learning_rate: float = 2e-5,
    epochs: int = 5,
    batch_size: int = 4,
    output_dir: str = "./owlv2_output",
):
    """
    Train OWLv2 on a JSONL dataset.
    """
    config = OWLv2Configuration(
        model_name=model_name,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    from transformers import OwlViTProcessor
    processor = OwlViTProcessor.from_pretrained(model_name)
    train_dataset = OWLv2JSONLDataset(jsonl_path, images_dir, processor)
    train(config, train_dataset)

@owlv2_app.command()
def infer(
    model_path: str,
    processor_path: str,
    image_path: str,
    prompts: str,
    device: str = "cuda",
):
    """
    Run inference with a fine-tuned OWLv2 model.
    """
    prompt_list = [p.strip() for p in prompts.split(",")]
    result = run_inference(model_path, processor_path, image_path, prompt_list, device=device)
    print(result)
