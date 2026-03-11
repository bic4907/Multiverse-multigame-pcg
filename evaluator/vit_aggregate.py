import pandas as pd
import json
import os


def process_single_instruction(single_df: pd.DataFrame) -> dict:
    """
    Processes single instruction data and returns results (wandb hierarchy)

    Args:
        single_df: Single instruction DataFrame

    Returns:
        dict: Per-game scores and overall score info (hierarchized with / separator)
    """
    result = {}

    # Average score per game
    game_scores = single_df.groupby('game')['score'].agg(['mean', 'count'])
    for game, row in game_scores.iterrows():
        result[f'vit-single/{game}'] = round(float(row['mean']), 4)

    # Overall average score
    result['vit-single/overall/score'] = round(float(single_df['score'].mean()), 4)

    return result


def process_blended_instruction(blended_df: pd.DataFrame) -> dict:
    """
    Processes blended instruction data and returns results (wandb hierarchy)

    Args:
        blended_df: Blended instruction DataFrame

    Returns:
        dict: Per-game, per-ratio, and overall score info (hierarchized with / separator)
    """
    result = {}

    # 1. Per-game scores and Score Coefficient
    game_blend_summary = blended_df.groupby(['game_a', 'game_b']).agg({
        'score': ['mean', 'count']
    })

    for (game_a, game_b), row in game_blend_summary.iterrows():
        key = f"{game_a}_{game_b}"
        group = blended_df[(blended_df['game_a'] == game_a) & (blended_df['game_b'] == game_b)]

        # Correlation between ratio_a and score_ac
        corr_a = group['ratio_a'].corr(group['score_ac'])
        # Correlation between ratio_b and score_bc
        corr_b = group['ratio_b'].corr(group['score_bc'])

        result[f'vit-blend-detail/{key}/score_ac'] = round(float(group['score_ac'].mean()), 4)
        result[f'vit-blend-detail/{key}/score_bc'] = round(float(group['score_bc'].mean()), 4)
        result[f'vit-blend-detail/{key}/score'] = round(float(row[('score', 'mean')]), 4)
        result[f'vit-blend-detail/{key}/corr_a'] = round(float(corr_a), 4)
        result[f'vit-blend-detail/{key}/corr_b'] = round(float(corr_b), 4)

    # 2. Per-ratio scores
    for (ratio_a, ratio_b), group in blended_df.groupby(['ratio_a', 'ratio_b']):
        key = f"{ratio_a}_{ratio_b}"

        result[f'vit-blend/ratio_{key}/score_ac'] = round(float(group['score_ac'].mean()), 4)
        result[f'vit-blend/ratio_{key}/score_bc'] = round(float(group['score_bc'].mean()), 4)
        result[f'vit-blend/ratio_{key}/score'] = round(float(group['score'].mean()), 4)

    # 3. Overall scores and Score Coefficient
    result['vit-blend/overall/score'] = round(float(blended_df['score'].mean()), 4)

    # Overall coefficient across all data
    overall_corr_a = blended_df['ratio_a'].corr(blended_df['score_ac'])
    overall_corr_b = blended_df['ratio_b'].corr(blended_df['score_bc'])
    result['vit-blend/overall/corr_a'] = round(float(overall_corr_a), 4)
    result['vit-blend/overall/corr_b'] = round(float(overall_corr_b), 4)
    result['vit-blend/overall/corr'] = round(float((overall_corr_a + overall_corr_b) / 2), 4)

    return result

def process_text_blended_instruction(blended_df: pd.DataFrame) -> dict:
    """
    Processes text-blended instruction data and returns results (wandb hierarchy)

    Args:
        blended_df: Text-blended instruction DataFrame

    Returns:
        dict: Per-game, per-blend-type, and overall score info (hierarchized with / separator)
    """
    result = {}

    # 1. Per-game scores and Score Coefficient
    game_blend_summary = blended_df.groupby(['game_a', 'game_b']).agg({
        'score': ['mean', 'count']
    })

    for (game_a, game_b), row in game_blend_summary.iterrows():
        key = f"{game_a}_{game_b}"
        group = blended_df[(blended_df['game_a'] == game_a) & (blended_df['game_b'] == game_b)]

        result[f'vit-blend-detail/{key}/score_ac'] = round(float(group['score_ac'].mean()), 4)
        result[f'vit-blend-detail/{key}/score_bc'] = round(float(group['score_bc'].mean()), 4)
        result[f'vit-blend-detail/{key}/score'] = round(float(row[('score', 'mean')]), 4)

    # 2. Per-blend-type scores
    for blend_type, group in blended_df.groupby(['blend_type']):

        result[f'vit-blend/{blend_type}/score_ac'] = round(float(group['score_ac'].mean()), 4)
        result[f'vit-blend/{blend_type}/score_bc'] = round(float(group['score_bc'].mean()), 4)
        result[f'vit-blend/{blend_type}/score'] = round(float(group['score'].mean()), 4)

    # 3. Overall scores and Score Coefficient
    result['vit-blend/overall/score'] = round(float(blended_df['score'].mean()), 4)

    return result

if __name__ == '__main__':
    # Get the directory path of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Set CSV file paths
    single_csv_path = os.path.join(current_dir, 'samples', 'single_instruction.csv')
    blended_csv_path = os.path.join(current_dir, 'samples', 'blended_instruction.csv')

    # Load data
    single_df = pd.read_csv(single_csv_path)
    blended_df = pd.read_csv(blended_csv_path)

    # Process Single Instruction
    print("\n" + "=" * 60)
    print("Single Instruction Results")
    print("=" * 60)

    single_result = process_single_instruction(single_df)
    print(json.dumps(single_result, indent=2, ensure_ascii=False))

    # Process Blended Instruction
    print("\n" + "=" * 60)
    print("Blended Instruction Results")
    print("=" * 60)

    blended_result = process_blended_instruction(blended_df)
    print(json.dumps(blended_result, indent=2, ensure_ascii=False))

    # Merge all results into a single JSON
    print("\n" + "=" * 60)
    print("Combined Results (JSON)")
    print("=" * 60)

    print(json.dumps(single_result, indent=2, ensure_ascii=False))
    print("\n" + "=" * 60)
    print("Combined Results (JSON)")
    print("=" * 60)
    print(json.dumps(blended_result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
