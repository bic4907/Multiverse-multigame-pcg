import os
import base64
import csv
from conf.config import PreProcessConfig
from openai import OpenAI


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_annotation(client, openai_model, image_path):
    image_b64 = encode_image(image_path)

    response = client.responses.create(
        model=openai_model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Classify this game map image into exactly one label.\n"
                            "Describe the distribution or location of objects in a map.\n"
                            "Do NOT describe colors.\n"
                            "Use these terms in this game: ground, breakable_block, question_block, goomba, pipe, coin, cannon\n"
                            "Output ONE label only. No explanation. Limit to 18 tokens.\n"
                            "Do not use any special symbols.\n"
                        )
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}"
                    }
                ]
            }
        ],
        max_output_tokens=20,
        temperature=0.2
    )

    annotation = response.output_text.strip()
    return annotation

def annotate_image(client, image_path, annotation_csv, image_exts, openai_model):
    rows = []

    for fname in sorted(os.listdir(image_path)):
        if not fname.lower().endswith(image_exts):
            continue

        image_file = os.path.join(image_path, fname)

        try:
            instruction = get_annotation(client, openai_model, image_file)
        except Exception as e:
            print(f"Failed: {fname} | {e}")
            instruction = "error"

        name_no_ext = os.path.splitext(fname)[0]
        print(f"{name_no_ext} -> {instruction}")
        rows.append([instruction, name_no_ext])

    with open(annotation_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instruction", "filename"])
        writer.writerows(rows)

    print(f"\nSaved CSV to {annotation_csv}")


def main(config):
    client = OpenAI(
        api_key=config.openai_api_key
    )

    annotate_image(
        client=client,
        image_path=config.image_path, 
        annotation_csv=config.annotation_csv,
        image_exts=config.image_exts,
        openai_model=config.openai_model
    )


if __name__ == "__main__":
    config = PreProcessConfig()
    main(config)
