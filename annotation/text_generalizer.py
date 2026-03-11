import pandas as pd
from conf.config import PreProcessConfig

def generalize_text(input_csv_path, output_csv_path):

    REPLACE_MAP = {
        "bats": "enemies",
        "bat": "enemy",
        "monsters": "enemies",
        "monster": "enemy",
        "goonmbas": "enemies",
        "goomba": "enemy",
         
        "waterfloor": "element_with_floor",
        "waterblock": "element_with_block",
        "water_pool": "element",
        "water": "element",
        "stair": "climbable_block",
        "rope": "climbable_block",
        "ladder": "climbable_block",
        "gold": "collectable_block",
        "question_block": "breakable_block",

        "coin": "collectable_block",
        "cannon": "hazard_block"
    }

    df = pd.read_csv(input_csv_path)

    def replace_instruction(text: str) -> str:
        for src, tgt in REPLACE_MAP.items():
            text = text.replace(src, tgt)
        return text

    df["instruction"] = df["instruction"].apply(replace_instruction)

    df.to_csv(output_csv_path, index=False)

    print(f"Saved to {output_csv_path}")

def main(config):
    generalize_text(
        input_csv_path=config.specific_annotation_csv,
        output_csv_path=config.general_annotation_csv
    )

if __name__ == "__main__":
    config = PreProcessConfig()
    main(config)
