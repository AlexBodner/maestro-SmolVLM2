from dataclasses import dataclass
from typing import Optional
import os
import json

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor, TrainingArguments, Trainer


@dataclass
class OWLv2Configuration:
    model_name: str = "google/owlv2-base-patch16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 2e-5
    epochs: int = 5
    batch_size: int = 4
    output_dir: str = "./owlv2_output"
    # Add more config options as needed


def train(config: OWLv2Configuration, train_dataset, val_dataset=None):
    """
    Fine-tune OWLv2 on the provided dataset.
    Args:
        config: OWLv2Configuration
        train_dataset: torch.utils.data.Dataset
        val_dataset: torch.utils.data.Dataset or None
    """
    model = OwlViTForObjectDetection.from_pretrained(config.model_name).to(config.device)
    processor = OwlViTProcessor.from_pretrained(config.model_name)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        evaluation_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="epoch",
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        data_collator=None,  # You may need a custom collator for object detection
    )

    trainer.train()
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)
    print(f"[OWLv2] Training complete. Model saved to {config.output_dir}")
    return model, processor


# Dataset handling utility (for JSONL referencing images)
class OWLv2JSONLDataset(Dataset):
    def __init__(self, jsonl_path, images_dir, processor, mode="RGB"):
        self.entries = []
        with open(jsonl_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.entries.append(entry)
        self.images_dir = images_dir
        self.processor = processor
        self.mode = mode

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = entry["image"] if os.path.isabs(entry["image"]) else os.path.join(self.images_dir, entry["image"])
        image = Image.open(image_path).convert(self.mode)
        # You may need to adapt this for N-channel/high bit-depth TIFFs
        # Prepare the target/labels as required by OWLv2
        # Example: {"boxes": ..., "labels": ...}
        # For now, just return image and entry
        return {"image": image, "labels": entry.get("labels", None), "boxes": entry.get("boxes", None)}
