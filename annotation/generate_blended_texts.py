import json
import os
import hydra
from tqdm import tqdm
import pandas as pd
import random
from openai import OpenAI

from conf.config import TextBlendEvalConfig
from conf.initializer.rollout import init_config
from data_loader import make_dataloaders
from instruct.utils import get_csv_path
from evaluator.blend_dataset import make_text_blender_dataloader


def make_blend_pairs(data_loader, output_csv_path, max_data_length=2009, seed=42):

    blend_loader = make_text_blender_dataloader(
        data_loader=data_loader,
        max_data_length=max_data_length,
    )

    rows = []

    for batch in tqdm(blend_loader, desc="collect-blend-pairs"):
        batch_size = len(batch["text_a"])

        for i in range(batch_size):
            rows.append({
                "game_a": batch["game_a"][i],
                "text_a": batch["text_a"][i],
                "level_id_a": batch["level_id_a"][i],

                "game_b": batch["game_b"][i],
                "text_b": batch["text_b"][i],
                "level_id_b": batch["level_id_b"][i],
            })

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

    df.to_csv(output_csv_path, index=False)

    print(f"Saved {len(df)} blend pairs → {output_csv_path}")

    return df


# Prompt Builder
def build_prompt(text_a: str, text_b: str, blend_type: str):

    if blend_type == "mix":
        if random.random() < 0.5:
            text_a, text_b = text_b, text_a
            
        instruction = (
            "Synthesize the two descriptions into a unified spatial narrative. "
            "Blend overlapping or related elements when possible, and reorganize "
            "the scene to read as a naturally designed environment. "
            "Avoid comma-separated listing."
        )

    elif blend_type == "a_base":
        instruction = (
            "Text A is the base instruction. "
            "Text B is an additional modifier instruction. "
            "Rewrite Text A by incorporating elements from Text B."
        )

    elif blend_type == "b_base":
        text_a, text_b = text_b, text_a
        instruction = (
            "Text A is the base instruction. "
            "Text B is an additional modifier instruction. "
            "Rewrite Text A by incorporating elements from Text B."
        )

    else:
        raise ValueError(f"Unsupported blend type: {blend_type}")

    return (
        f"{instruction}\n\n"
        f"Text A: {text_a}\n"
        f"Text B: {text_b}\n\n"
        "Return only the blended sentence. Do not use any special symbols."
    )

# Main Batch Generation
def create_batch_requests(df_pairs, model):

    blend_types = ["mix", "a_base", "b_base"]
    requests = []

    for idx, row in tqdm(df_pairs.iterrows(), total=len(df_pairs)):

        text_a = row["text_a"]
        text_b = row["text_b"]

        for bt in blend_types:

            prompt = build_prompt(text_a, text_b, bt)

            custom_id = f"{idx}__{bt}"

            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": [
                        {
                            "role": "system",
                            "content": "You are a semantic blending assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.2
                }
            })

    return requests

def generate_texts(config, client):
    # ---- dataloaders ----
    train_loader, val_loader = make_dataloaders(
        get_csv_path(config.instruction_csv),
        batch_size=config.batch_size,
        game_list=config.game_list,
        level_size=config.level_size,
        active_losses=config.active_losses,
        sample_ratio=config.sample_ratio,
        val_ratio=config.val_ratio,
        clip_model=config.clip_model,
        seed=config.dataset_split_seed,
        device=config.device,
        trainset_game=getattr(config, "trainset_game", None),
    )

    save_dir = os.path.join(
        "evaluator",
        "text_blend",
    )
    os.makedirs(save_dir, exist_ok=True)

    # ---- Generate blend pairs ----
    blend_pairs_path = os.path.join(save_dir, "blended_pairs.csv")
    df_pairs = make_blend_pairs(
        data_loader=val_loader,
        output_csv_path=blend_pairs_path,
        max_data_length=2009,
    )
    
    # ---- Create batch requests ----
    requests = create_batch_requests(df_pairs, config.openai_model)

    batch_file_path = os.path.join(save_dir, "blend_batch.jsonl")

    with open(batch_file_path, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(requests)} batch requests")

    # ---- Upload to OpenAI ----
    batch_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    print("Batch job created:", batch_job.id)
    print("Check status with this ID later.")
    print("When completed, download results using process_batch_output.py")


@hydra.main(version_base="1.3", config_path="conf", config_name="eval_textblend")
def main(config: TextBlendEvalConfig):
    init_config(config)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    generate_texts(config, client)


if __name__ == "__main__":
    main()