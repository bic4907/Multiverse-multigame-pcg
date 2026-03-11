import pandas as pd
import json
import os


def process_single_instruction(single_df: pd.DataFrame) -> dict:
    """
    Single instruction 데이터를 처리하여 결과를 반환 (wandb 계층 구조)

    Args:
        single_df: Single instruction DataFrame

    Returns:
        dict: 게임별 스코어와 전체 스코어 정보 (/ 구분자로 계층화)
    """
    result = {}

    # 게임별 평균 스코어
    game_scores = single_df.groupby('game')['score'].agg(['mean', 'count'])
    for game, row in game_scores.iterrows():
        result[f'vit-single/{game}'] = round(float(row['mean']), 4)

    # 전체 평균 스코어
    result['vit-single/overall/score'] = round(float(single_df['score'].mean()), 4)

    return result


def process_blended_instruction(blended_df: pd.DataFrame) -> dict:
    """
    Blended instruction 데이터를 처리하여 결과를 반환 (wandb 계층 구조)

    Args:
        blended_df: Blended instruction DataFrame

    Returns:
        dict: 게임별, ratio별, 전체 스코어 정보 (/ 구분자로 계층화)
    """
    result = {}

    # 1. 게임별 스코어 및 Score Coefficient
    game_blend_summary = blended_df.groupby(['game_a', 'game_b']).agg({
        'score': ['mean', 'count']
    })

    for (game_a, game_b), row in game_blend_summary.iterrows():
        key = f"{game_a}_{game_b}"
        group = blended_df[(blended_df['game_a'] == game_a) & (blended_df['game_b'] == game_b)]

        # ratio_a와 score_ac의 상관관계
        corr_a = group['ratio_a'].corr(group['score_ac'])
        # ratio_b와 score_bc의 상관관계
        corr_b = group['ratio_b'].corr(group['score_bc'])

        result[f'vit-blend-detail/{key}/score_ac'] = round(float(group['score_ac'].mean()), 4)
        result[f'vit-blend-detail/{key}/score_bc'] = round(float(group['score_bc'].mean()), 4)
        result[f'vit-blend-detail/{key}/score'] = round(float(row[('score', 'mean')]), 4)
        result[f'vit-blend-detail/{key}/corr_a'] = round(float(corr_a), 4)
        result[f'vit-blend-detail/{key}/corr_b'] = round(float(corr_b), 4)

    # 2. Ratio별 스코어
    for (ratio_a, ratio_b), group in blended_df.groupby(['ratio_a', 'ratio_b']):
        key = f"{ratio_a}_{ratio_b}"

        result[f'vit-blend/ratio_{key}/score_ac'] = round(float(group['score_ac'].mean()), 4)
        result[f'vit-blend/ratio_{key}/score_bc'] = round(float(group['score_bc'].mean()), 4)
        result[f'vit-blend/ratio_{key}/score'] = round(float(group['score'].mean()), 4)

    # 3. 전체 스코어 및 전체 Score Coefficient
    result['vit-blend/overall/score'] = round(float(blended_df['score'].mean()), 4)

    # 전체 데이터에 대한 coefficient
    overall_corr_a = blended_df['ratio_a'].corr(blended_df['score_ac'])
    overall_corr_b = blended_df['ratio_b'].corr(blended_df['score_bc'])
    result['vit-blend/overall/corr_a'] = round(float(overall_corr_a), 4)
    result['vit-blend/overall/corr_b'] = round(float(overall_corr_b), 4)
    result['vit-blend/overall/corr'] = round(float((overall_corr_a + overall_corr_b) / 2), 4)

    return result

def process_text_blended_instruction(blended_df: pd.DataFrame) -> dict:
    """
    Text-blended instruction 데이터를 처리하여 결과를 반환 (wandb 계층 구조)

    Args:
        blended_df: Text-blended instruction DataFrame

    Returns:
        dict: 게임별, blend type별, 전체 스코어 정보 (/ 구분자로 계층화)
    """
    result = {}

    # 1. 게임별 스코어 및 Score Coefficient
    game_blend_summary = blended_df.groupby(['game_a', 'game_b']).agg({
        'score': ['mean', 'count']
    })

    for (game_a, game_b), row in game_blend_summary.iterrows():
        key = f"{game_a}_{game_b}"
        group = blended_df[(blended_df['game_a'] == game_a) & (blended_df['game_b'] == game_b)]

        result[f'vit-blend-detail/{key}/score_ac'] = round(float(group['score_ac'].mean()), 4)
        result[f'vit-blend-detail/{key}/score_bc'] = round(float(group['score_bc'].mean()), 4)
        result[f'vit-blend-detail/{key}/score'] = round(float(row[('score', 'mean')]), 4)

    # 2. blend type별 스코어
    for blend_type, group in blended_df.groupby(['blend_type']):

        result[f'vit-blend/{blend_type}/score_ac'] = round(float(group['score_ac'].mean()), 4)
        result[f'vit-blend/{blend_type}/score_bc'] = round(float(group['score_bc'].mean()), 4)
        result[f'vit-blend/{blend_type}/score'] = round(float(group['score'].mean()), 4)

    # 3. 전체 스코어 및 전체 Score Coefficient
    result['vit-blend/overall/score'] = round(float(blended_df['score'].mean()), 4)

    return result

if __name__ == '__main__':
    # 현재 파일의 디렉토리 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # CSV 파일 경로 설정
    single_csv_path = os.path.join(current_dir, 'samples', 'single_instruction.csv')
    blended_csv_path = os.path.join(current_dir, 'samples', 'blended_instruction.csv')

    # 데이터 로드
    single_df = pd.read_csv(single_csv_path)
    blended_df = pd.read_csv(blended_csv_path)

    # Single Instruction 처리
    print("\n" + "=" * 60)
    print("Single Instruction 결과")
    print("=" * 60)

    single_result = process_single_instruction(single_df)
    print(json.dumps(single_result, indent=2, ensure_ascii=False))

    # Blended Instruction 처리
    print("\n" + "=" * 60)
    print("Blended Instruction 결과")
    print("=" * 60)

    blended_result = process_blended_instruction(blended_df)
    print(json.dumps(blended_result, indent=2, ensure_ascii=False))

    # 전체 결과를 하나의 JSON으로 합치기
    print("\n" + "=" * 60)
    print("전체 결과 (Combined JSON)")
    print("=" * 60)

    # combined_result = {
    #     'single_instruction': single_result,
    #     'blended_instruction': blended_result
    # }
    print(json.dumps(single_result, indent=2, ensure_ascii=False))
    print("\n" + "=" * 60)
    print("전체 결과 (Combined JSON)")
    print("=" * 60)
    print(json.dumps(blended_result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
