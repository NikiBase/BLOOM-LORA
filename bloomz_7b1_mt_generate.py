# pip install -q transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.read import read_file


def evaluation(source_file, src_lang, tgt_lang):
    sources = read_file(source_file)
    checkpoint = "bigscience/bloomz-7b1-mt"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype="auto", device_map="auto"
    )
    with open(f"translation_{src_lang}_{tgt_lang}_hyp.txt", "w", encoding="utf8") as f:
        for src in sources:
            prompt = f"Translate to {tgt_lang}: {src}"
            inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(inputs, max_new_tokens=1024)
            f.write(f"{str(tokenizer.decode(outputs[0])).strip()}\n")


if __name__ == "__main__":
    evaluation(
        source_file="data/official_test.en",
        src_lang="English",
        tgt_lang="Spanish",
    )
