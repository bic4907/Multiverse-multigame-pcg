import numpy as np
from os.path import join, dirname, abspath
from PIL import Image
import pandas as pd

TILE_IM_ROOT = abspath(join(dirname(__file__), 'tile_ims'))
TILE_CSV_PATH = abspath(join(dirname(__file__), 'tiles.csv'))

def tile_img_mapping():
    df = pd.read_csv(TILE_CSV_PATH)

    tile_img_mapping = {}

    for _, row in df.iterrows():
        tile_id = int(row["tile_id"])
        file_name = row["file_name"]
        game = row["game"]

        img_path = join(TILE_IM_ROOT, game, f"{file_name}.png")

        tile_img_mapping[tile_id] = Image.open(img_path).convert("RGBA")

    return tile_img_mapping

TILE_IMG_MAPPING = tile_img_mapping()

def render_level(
    array: np.ndarray,
    tile_size: int = 16,
    return_numpy: bool = True,
):
    """
    Render a level array (single or batch) into image(s).

    Args:
        array:
            (H, W) or (B, H, W)
        tile_size:
            Tile pixel size
        return_numpy:
            If True, return np.ndarray
            If False, return PIL.Image or List[PIL.Image]

    Returns:
        np.ndarray | PIL.Image | List[PIL.Image]
    """

    # --------------------------------------------------
    # Normalize input to (B, H, W)
    # --------------------------------------------------
    if array.ndim == 2:
        array = array[None, ...]  # (1, H, W)
        squeeze_batch = True
    elif array.ndim == 3:
        squeeze_batch = False
    else:
        raise ValueError(f"Expected (H,W) or (B,H,W), got {array.shape}")

    B, H, W = array.shape
    rendered_imgs = []

    # --------------------------------------------------
    # Render each level
    # --------------------------------------------------
    for b in range(B):
        level = array[b]
        img = np.zeros((H * tile_size, W * tile_size, 4), dtype=np.uint8)

        for y in range(H):
            for x in range(W):
                val = int(level[y, x])

                if val not in TILE_IMG_MAPPING:
                    continue

                tile_img = TILE_IMG_MAPPING[val].resize(
                    (tile_size, tile_size),
                    resample=Image.NEAREST,
                )
                tile_np = np.asarray(tile_img, dtype=np.uint8)

                y0, y1 = y * tile_size, (y + 1) * tile_size
                x0, x1 = x * tile_size, (x + 1) * tile_size
                img[y0:y1, x0:x1, :] = tile_np

        rendered_imgs.append(img)

    # --------------------------------------------------
    # Return format
    # --------------------------------------------------
    if return_numpy:
        rendered_imgs = np.stack(rendered_imgs, axis=0)  # (B, H*T, W*T, 4)
        if squeeze_batch:
            return rendered_imgs[0]
        return rendered_imgs

    else:
        pil_imgs = [Image.fromarray(img, mode="RGBA") for img in rendered_imgs]
        if squeeze_batch:
            return pil_imgs[0]
        return pil_imgs


if __name__ == '__main__':
    sample_level = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 1, 3, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 2, 1],
        [1, 1, 3, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.int32)

    img = render_level(sample_level, tile_size=32, return_numpy=False)
    img.show()

    pass