import os
import time
from functools import lru_cache

import torch
from datasets import load_dataset
import transformers

from utils.s3 import zip_n_store


assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import AutoTokenizer, BloomForCausalLM
import json
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from generate_bloom import generate


@lru_cache
def get_config(file: str = "train_config.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["gradient_accumulation_steps"] = data["batch_size"] // data["micro_batch_size"]
    return data


"""
The folling models are available for fine-tunning:
    "bigscience/bloom-560m"
    "bigscience/bloom-1b1"
    "bigscience/bloom-1b7"
    "bigscience/bloom-3b"
    "bigscience/bloom-7b1"
    "bigscience/bloom" # for 176B parameters
"""


def train():
    # Load training parameters
    data = get_config()
    model_name = data["model_name"]
    lora_r = data["lora_r"]
    lora_alpha = data["lora_alpha"]
    lora_dropout = data["lora_dropout"]
    data_path = data["data_path"]
    val_set_size = data["val_set_size"]
    cutoff_len = data["cutoff_len"]
    micro_batch_size = data["micro_batch_size"]
    gradient_accumulation_steps = data["gradient_accumulation_steps"]
    epochs = data["epochs"]
    lr = data["lr"]
    output_dir = data["output_dir"]
    ddp = data["ddp"]
    resume_from_checkpoint = data["resume_from_checkpoint"]
    s3_bucket = data["s3_bucket"]

    model = BloomForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=None,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files=data_path)

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = train_val["train"]
    val_data = train_val["test"]

    def generate_and_tokenize_prompt(data_point):
        # This function masks out the labels for the input,
        # so that our loss is computed only on the response.
        user_prompt = (
            (
                f"""### Instruction: {data_point["instruction"]}\n\n### Input: {data_point["input"]}\n\n### Response: """
            )
            if data_point["input"]
            else (f"""### Instruction: {data_point["instruction"]}\n\n### Response: """)
        )
        len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=cutoff_len + 1,
                    padding="max_length",
                )["input_ids"]
            )
            - 1
        )  # no eos token
        full_tokens = tokenizer(
            user_prompt + data_point["output"],
            truncation=True,
            max_length=cutoff_len + 1,
            padding="max_length",
        )["input_ids"][:-1]
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
    val_data = val_data.shuffle().map(generate_and_tokenize_prompt)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2":
        model = torch.compile(model)
    start_train = time.time()
    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )  # if resume, choose True, else False
    end_train = time.time() - start_train
    with open(
        os.path.join(output_dir, f"train_timing_{output_dir}.txt"), "w", encoding="utf8"
    ) as f:
        f.write(f"Time to train: {end_train}")
    model.save_pretrained(output_dir)

    # zip_n_store(output_dir, s3_bucket, output_dir + ".zip")


if __name__ == "__main__":
    train()
