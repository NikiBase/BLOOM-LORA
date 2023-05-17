import json



def read_json(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def read_file(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        return f.readlines()
