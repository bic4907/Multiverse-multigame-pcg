import json
import os
import random
import hydra
import torch
from tqdm import tqdm
import pandas as pd
from openai import OpenAI

from conf.config import TextBlendEvalConfig
from conf.initializer.rollout import init_config
from data_loader import make_dataloaders
from instruct.utils import get_csv_path
from evaluator.blend_dataset import make_text_blender_dataloader


def extract_output_text(response_obj):
    try:
        return response_obj["response"]["body"]["output"][0]["content"][0]["text"].strip()
    except Exception:
        return ""


def download_texts(config, client):
    print("Retrieving batch job...")
    batch = client.batches.retrieve(config.batch_id)

    if batch.status != "completed":
        print("Batch not completed yet. Current status:", batch.status)
        return

    print("Downloading output file...")
    result_file_id = batch.output_file_id
    content = client.files.content(result_file_id)

    lines = content.text.strip().split("\n")
    for line in lines[:5]:
        print(line)
    results = {}

    print("Parsing batch results...")

    for line in tqdm(lines):
        obj = json.loads(line)
        print(obj.keys())
        print(obj["custom_id"])
        custom_id = obj["custom_id"]
        text_output = extract_output_text(obj)

        results[custom_id] = text_output

    save_dir = os.path.join(
        "evaluator",
        "text_blend",
    )
    os.makedirs(save_dir, exist_ok=True)

    blend_pairs_path = os.path.join(save_dir, "blended_pairs.csv")

    df_pairs = pd.read_csv(blend_pairs_path)

    blended_rows = []

    print("Reconstructing blended text dataframe...")

    for idx, row in tqdm(df_pairs.iterrows(), total=len(df_pairs)):

        game_a = row["game_a"]
        text_a = row["text_a"]
        level_id_a = row["level_id_a"]

        game_b = row["game_b"]
        text_b = row["text_b"]
        level_id_b = row["level_id_b"]

        # concat
        texts = [text_a, text_b]
        random.shuffle(texts)

        concat_text = f"{texts[0]} {texts[1]}"
        blended_rows.append({
            "row_id": idx,
            "game_a": game_a,
            "text_a": text_a,
            "level_id_a": level_id_a,

            "game_b": game_b,
            "text_b": text_b,
            "level_id_b": level_id_b,

            "blend_type": "concat",
            "text_c": concat_text,
        })

        for bt in ["mix", "a_base", "b_base"]:

            custom_id = f"{idx}__{bt}"
            blended_text = results.get(custom_id, "")

            blended_rows.append({
                "row_id": idx,
                "game_a": game_a,
                "text_a": text_a,
                "level_id_a": level_id_a,

                "game_b": game_b,
                "text_b": text_b,
                "level_id_b": level_id_b,

                "blend_type": bt,
                "text_c": blended_text,
            })

    df_blended = pd.DataFrame(blended_rows)

    output_path = os.path.join(config.blended_instuction_csv)

    df_blended.to_csv(output_path, index=False)

    print("Saved blended texts to:", output_path)


@hydra.main(version_base="1.3", config_path="conf", config_name="eval_textblend")
def main(config: TextBlendEvalConfig):
    init_config(config)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    download_texts(config, client)

if __name__ == "__main__":
    main()