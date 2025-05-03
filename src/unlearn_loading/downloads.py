#!/usr/bin/env python3

import os
import pandas as pd
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
import torch

def download_model(
    path="semeval25-unlearning-model",
    model_type="OLMo"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Download and load the specified model.
    
    Args:
        path: Path to save the model
        model_type: Type of model to load ("OLMo" for OLMo-7B or "Llama-2-7b-chat" for TOFU Llama model)
    """
    if model_type == "Llama-2-7b-chat":
        print("Using Llama-2-7b-chat model")
        # Load tokenizer from Meta's base model
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_fast=True,
            trust_remote_code=True
        )
        # Set padding token to be the same as EOS token
        tokenizer.pad_token = tokenizer.eos_token
        print("Loading model from local directory:", path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        return model, tokenizer
    
    # Default OLMo model
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning",
            local_dir=path,
        )
    return AutoModelForCausalLM.from_pretrained(path), AutoTokenizer.from_pretrained(
        "allenai/OLMo-7B-0724-Instruct-hf"
    )


def download_model_1B(
    path="semeval25-unlearning-1B-model",
    model_type="1B"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Download and load the specified 1B model.
    
    Args:
        path: Path to save the model
        model_type: Type of model to load ("1B" for OLMo-1B or "Llama-3.2-1B" for Meta's Llama model)
    """
    if model_type == "Llama-3.2-1B":
        print("Using Llama-3.2-1B model")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            trust_remote_code=True
        )
        # Set padding token to be the same as EOS token
        tokenizer.pad_token = tokenizer.eos_token
        return AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            device_map="auto",
            trust_remote_code=True
        ), tokenizer
    
    # Default OLMo-1B model
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning",
            local_dir=path,
        )

    return AutoModelForCausalLM.from_pretrained(path), AutoTokenizer.from_pretrained(
        "allenai/OLMo-1B-0724-hf"
    )


def download_datasets(
    path="semeval25-unlearning-data",
    val_split=0.1,
    seed=42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download and process TOFU datasets"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        os.makedirs(path + "/data", exist_ok=True)

        # Load datasets
        forget_ds = load_dataset("locuslab/TOFU", "forget01")["train"]
        retain_ds = load_dataset("locuslab/TOFU", "retain99")["train"]

        # Split into train/val
        forget_splits = forget_ds.train_test_split(test_size=val_split, seed=seed)
        retain_splits = retain_ds.train_test_split(test_size=val_split, seed=seed)

        # Transform format and add IDs
        forget_train = forget_splits["train"].map(
            lambda x: transform_dataset_format(x, start_id=0, is_forget=True),
            batched=True,
            remove_columns=forget_splits["train"].column_names
        )
        forget_val = forget_splits["test"].map(
            lambda x: transform_dataset_format(x, start_id=len(forget_train), is_forget=True),
            batched=True,
            remove_columns=forget_splits["test"].column_names
        )
        retain_train = retain_splits["train"].map(
            lambda x: transform_dataset_format(x, start_id=len(forget_train) + len(forget_val), is_forget=False),
            batched=True,
            remove_columns=retain_splits["train"].column_names
        )
        retain_val = retain_splits["test"].map(
            lambda x: transform_dataset_format(x, start_id=len(forget_train) + len(forget_val) + len(retain_train), is_forget=False),
            batched=True,
            remove_columns=retain_splits["test"].column_names
        )

        # Convert to pandas and save
        forget_train_df = pd.DataFrame(forget_train)
        forget_validation_df = pd.DataFrame(forget_val)
        retain_train_df = pd.DataFrame(retain_train)
        retain_validation_df = pd.DataFrame(retain_val)

        # Save each split
        forget_train_df.to_parquet(
            os.path.join(path, "data/forget_train-00000-of-00001.parquet"),
            engine="pyarrow",
        )
        forget_validation_df.to_parquet(
            os.path.join(path, "data/forget_validation-00000-of-00001.parquet"),
            engine="pyarrow",
        )
        retain_train_df.to_parquet(
            os.path.join(path, "data/retain_train-00000-of-00001.parquet"),
            engine="pyarrow",
        )
        retain_validation_df.to_parquet(
            os.path.join(path, "data/retain_validation-00000-of-00001.parquet"),
            engine="pyarrow",
        )

    # Load from cached parquet files
    retain_train_df = pd.read_parquet(
        os.path.join(path, "data/retain_train-00000-of-00001.parquet"),
        engine="pyarrow",
    )
    retain_validation_df = pd.read_parquet(
        os.path.join(path, "data/retain_validation-00000-of-00001.parquet"),
        engine="pyarrow",
    )
    forget_train_df = pd.read_parquet(
        os.path.join(path, "data/forget_train-00000-of-00001.parquet"),
        engine="pyarrow",
    )
    forget_validation_df = pd.read_parquet(
        os.path.join(path, "data/forget_validation-00000-of-00001.parquet"),
        engine="pyarrow",
    )
    return retain_train_df, retain_validation_df, forget_train_df, forget_validation_df


def main(args):
    download_model(os.path.join(args.path, "semeval25-unlearning-model"), model_type="Llama-2-7b-chat")
    download_model_1B(
        os.path.join(args.path, "semeval25-unlearning-1B-model")
    )
    download_datasets(os.path.join(args.path, "semeval25-unlearning-data"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", default=".", type=str, help="Path to save the downloaded files."
    )
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
