import os
import json
import numpy as np
from conf.config import PreProcessConfig


def load_tile_mapping(json_path, start_num):
    with open(json_path, "r") as f:
        data = json.load(f)

    tiles = data["tiles"]
    tile_to_id = {ch: idx for idx, ch in enumerate(tiles.keys(), start=start_num)}
    return tile_to_id


def load_level_text(txt_path):
    with open(txt_path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    height = len(lines)
    width = max(len(line) for line in lines)

    grid = [list(line) for line in lines]
    return grid, height, width


def level_to_numpy(text_path, tile_to_id):
    grid, h, w = load_level_text(text_path)

    level_np = np.zeros((h, w), dtype=np.int64)

    for y in range(h):
        for x in range(len(grid[y])):
            ch = grid[y][x]
            if ch not in tile_to_id:
                raise KeyError(f"{text_path}: unknown tile '{ch}' at ({y},{x})")
            level_np[y, x] = tile_to_id[ch]

    return level_np


def extract_patches(level_np, patch_size, stride):
    H, W = level_np.shape
    patches = []
    
    if stride > 0:
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                patch = level_np[y:y+patch_size, x:x+patch_size]

                if patch.size == 0:
                    continue

                patches.append(patch)
    else:
        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
                patch = level_np[y:y+patch_size, x:x+patch_size]
                patches.append(patch)

    return patches


def centering_and_padding(patch_np, window_size):
    h, w = patch_np.shape
    if h > window_size or w > window_size:
        raise ValueError(f"Patch ({h},{w}) larger than window {window_size}")

    # 1️window 생성
    window = np.zeros((window_size, window_size), dtype=patch_np.dtype)

    # 2️중앙 위치 계산
    top = (window_size - h) // 2
    left = (window_size - w) // 2
    bottom = top + h
    right = left + w

    # patch 덮어쓰기
    window[top:bottom, left:right] = patch_np

    # 위쪽 (row 0 복제)
    if top > 0:
        window[:top, left:right] = patch_np[0:1, :]

    # 아래쪽 (row -1 복제)
    if bottom < window_size:
        window[bottom:, left:right] = patch_np[-1:, :]

    # 왼쪽 (col 0 복제)
    if left > 0:
        window[:, :left] = window[:, left:left+1]

    # 오른쪽 (col -1 복제)
    if right < window_size:
        window[:, right:] = window[:, right-1:right]

    return window


def text_to_numpy(
    text_path,
    tile_json,
    raw_levels_path,
    start_num
):
    os.makedirs(raw_levels_path, exist_ok=True)

    tile_to_id = load_tile_mapping(tile_json, start_num)

    text_files = sorted([
        f for f in os.listdir(text_path)
        if f.endswith(".txt")
    ])

    print(f"Found {len(text_files)} level files")

    for fname in text_files:
        txt_path = os.path.join(text_path, fname)
        npy_name = os.path.splitext(fname)[0] + ".npy"
        raw_path = os.path.join(raw_levels_path, npy_name)

        try:
            level_np = level_to_numpy(txt_path, tile_to_id)
            print(level_np)
            np.save(raw_path, level_np)
            print(f"[Raw - OK] {fname} -> {npy_name}")

        except Exception as e:
            print(f"[Raw - FAIL] {fname}: {e}")


def raw_to_processed(
    raw_levels_path,
    processed_levels_path,
    processing_size,
    stride
):
    os.makedirs(processed_levels_path, exist_ok=True)

    npy_files = sorted([
        f for f in os.listdir(raw_levels_path)
        if f.endswith(".npy")
    ])

    print(f"Found {len(npy_files)} raw level files")

    for fname in npy_files:
        raw_path = os.path.join(raw_levels_path, fname)

        try:
            level_np = np.load(raw_path)

            patches = extract_patches(level_np, processing_size, stride)
            patch_name = ""
            for idx, patch_np in enumerate(patches, start=1):
                patch_np = centering_and_padding(patch_np, processing_size)

                patch_name = (
                    os.path.splitext(fname)[0]
                    + f"_patch{idx}.npy"
                )
                patch_path = os.path.join(processed_levels_path, patch_name)

                np.save(patch_path, patch_np)

            print(f"[Processed - OK] {fname} -> {patch_name}")

        except Exception as e:
            print(f"[Processed - FAIL] {fname}: {e}")


def process_levels(
    text_path,
    tile_json,
    raw_levels_path,
    processed_levels_path,
    processing_size,
    stride,
    start_num
):
    # text → raw_levels
    text_to_numpy(
        text_path=text_path,
        tile_json=tile_json,
        raw_levels_path=raw_levels_path,
        start_num=start_num
    )

    # raw_levels → processed_levels
    raw_to_processed(
        raw_levels_path=raw_levels_path,
        processed_levels_path=processed_levels_path,
        processing_size=processing_size,
        stride=stride
    )


def main(config):
    process_levels(
        text_path=config.text_path, 
        tile_json=config.tile_json, 
        raw_levels_path=config.raw_levels_path, 
        processed_levels_path=config.processed_levels_path,
        processing_size=config.processing_size,
        stride=config.stride,
        start_num=config.start_num
    )

if __name__ == "__main__":
    config = PreProcessConfig()
    main(config)