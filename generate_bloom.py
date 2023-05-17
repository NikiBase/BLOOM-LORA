from functools import lru_cache

import pandas as pd
import torch
from peft import PeftModel
import transformers
import gradio as gr

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    GenerationConfig,
)
from transformers.models.opt.modeling_opt import OPTDecoderLayer
import json


@lru_cache
def get_config(file: str = "generate_config.json"):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset(file):
    with open(file, "r", encoding="utf8") as f:
        return json.load(f)


def generate():

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
    data = get_config()

    model_path = data["model_path"]

    lora_weights = data["lora_weights"]
    load_in_8bit = data["load_in_8bit"]
    temperature = data["temperature"]
    top_p = (data["top_p"],)
    top_k = (data["top_k"],)
    num_beams = (data["num_beams"],)
    max_new_tokens = (data["max_new_tokens"],)
    test_path = data["test_path"]
    device = data["device"]
    if torch.cuda.is_available():
        device = "cuda"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass

    # Load test dataset
    test_data = get_dataset(test_path)

    if device == "cuda":
        model = BloomForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model, lora_weights, torch_dtype=torch.float16
        )
    elif device == "mps":
        model = BloomForCausalLM.from_pretrained(
            model_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = BloomForCausalLM.from_pretrained(
            model_path, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # ToDo: create a list of prompts or ask user via API
    def generate_prompt(instruction, input_=None):
        if input:
            return f"""### Instruction: {instruction}\n\n### Input: {input_}\n\n### Response: """
        else:
            return f"""### Instruction: {instruction}\n\n### Response:"""

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)
    output = []
    for data in test_data:
        instruction = data["instruction"]
        input_ = data["input"]
        prompt = generate_prompt(instruction, input_=input_)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        gen_output = tokenizer.decode(s).split("### Response:")[1].strip()
        output.append(
            {
                "instruction": instruction,
                "input": input_,
                "expected_output": data["output"],
                "generated_output": gen_output
            }
        )

    return output


if __name__ == "__main__":
    # testing code for readme
    generate()
    #
    #
    #
    # for instruction in [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500.",
    # ]:
    #     print("Instruction:", instruction)
    #     print("Response:", generate())
    #     print()
