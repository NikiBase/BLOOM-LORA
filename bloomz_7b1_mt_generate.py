# pip install -q transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.read import read_file


def evaluation(source_file, reference_file, src_lang, tgt_lang):
    sources = read_file(source_file)
    targets = read_file(reference_file)

    checkpoint = "bigscience/bloomz-7b1-mt"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype="auto", device_map="auto"
    )

    for src in sources:
        # prompt = (
        #     f"Translate the following {src_lang} text, which is delimited by triple backticks, to {tgt_lang}.\n"
        #     f"Return just the translation\n"
        #     f"```{src.strip()}```"
        # )
        prompt = f"Translate to {tgt_lang}: {src}"
        inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs)
        print(outputs)
        # print(tokenizer.decode(outputs[0]))
        exit()



if __name__ == "__main__":
    evaluation(
        source_file="data/official_test.en",
        reference_file="data/official_test.es",
        src_lang="English",
        tgt_lang="Spanish",
    )
