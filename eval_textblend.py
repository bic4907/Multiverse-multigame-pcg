import hydra
import torch
import os
import numpy as np
from transformers import CLIPTokenizer
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

from conf.config import TextBlendEvalConfig
from conf.initializer.rollout import init_config
from data_loader import make_dataloaders
from models.clip.clip_model import CLIPModel
from models.vae.ema_vqvae import EMA_VQVAE
from instruct.utils import get_csv_path
from utils.logger import get_logger, onehot_to_levels
from utils.path import get_log_dir
from utils.checkpoint import CheckpointManager, load_checkpoint_with_fallback
from evaluator.vitscore import ViTEvaluator
from evaluator import ThreePairComparisonSet

logger = get_logger(__file__)

@torch.no_grad()
def run_textblend_evaluation(
    data_loader,
    clip_model,
    vae_model,
    vit_evaluator,
    tokenizer,
    device: torch.device,
    df_blended: pd.DataFrame,
    output_csv_path: str,
):

    clip_model.eval()
    vae_model.eval()

    base_dataset = data_loader.dataset
    
    blended_levels = list()
    a_levels = list()
    b_levels = list()
    metadatas = list()
    extra_rows = list()

    for _, row in tqdm(df_blended.iterrows(), total=len(df_blended)):

        ga = row["game_a"]
        ta = row["text_a"]
        gb = row["game_b"]
        tb = row["text_b"]
        text_c = row["text_c"]
        blend_type = row["blend_type"]

        level_a = torch.argmax(
            base_dataset[row["level_id_a"]]["level"], dim=0
        ).cpu()

        level_b = torch.argmax(
            base_dataset[row["level_id_b"]]["level"], dim=0
        ).cpu()

        # ------------------------------------------------
        # text A/B embedding
        # ------------------------------------------------
        inputs_ab = tokenizer(
            [ta, tb],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        ab_embed = clip_model.text_encoder(
            inputs_ab["input_ids"],
            inputs_ab["attention_mask"],
        )
        ab_embed = F.normalize(ab_embed, dim=-1)

        ta_embed = ab_embed[0]
        tb_embed = ab_embed[1]

        # ------------------------------------------------
        # text C embedding
        # ------------------------------------------------
        inputs_c = tokenizer(
            [text_c],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        tc_embed = clip_model.text_encoder(
            inputs_c["input_ids"],
            inputs_c["attention_mask"],
        )
        tc_embed = F.normalize(tc_embed, dim=-1)[0]

        # ------------------------------------------------
        # Level generation
        # ------------------------------------------------
        logits = vae_model.sample(tc_embed.unsqueeze(0))
        generated = torch.argmax(logits, dim=1).cpu()
        gen_level = onehot_to_levels(generated)[0]

        blended_levels.append(gen_level)
        a_levels.append(level_a)
        b_levels.append(level_b)

        metadatas.append(
            (
                (ga, ta, 1.0),
                (gb, tb, 1.0),
            )
        )
        
        # ------------------------------------------------
        # similarity 계산
        # ------------------------------------------------
        sim_ac = F.cosine_similarity(ta_embed, tc_embed, dim=0).item()
        sim_bc = F.cosine_similarity(tb_embed, tc_embed, dim=0).item()


        extra_rows.append({
            "blend_type": blend_type,
            "text_c": text_c,
            "sim_ac": sim_ac,
            "sim_bc": sim_bc,
            "sim_diff": sim_ac - sim_bc,
        })

    blended_levels = torch.from_numpy(np.stack(blended_levels))
    a_levels = torch.from_numpy(np.stack(a_levels))
    b_levels = torch.from_numpy(np.stack(b_levels))
    
    # --------------------------------------------------------
    # ViT evaluation
    # --------------------------------------------------------
    evaluation_sets = [
        ThreePairComparisonSet(a, b, c, metadata=m)
        for a, b, c, m in zip(a_levels, b_levels, blended_levels, metadatas)
    ]
    _, three_pair = vit_evaluator.run(comparisons=evaluation_sets)
    df = three_pair.to_dataframe()

    df = df.drop(columns=["ratio_a", "ratio_b"])

    extra_df = pd.DataFrame(extra_rows)

    assert len(df) == len(extra_df), \
        f"Length mismatch: df={len(df)}, extra={len(extra_df)}"

    df = pd.concat(
        [df.reset_index(drop=True),
        extra_df.reset_index(drop=True)],
        axis=1
    )

    df = df[
        [
            "game_a",
            "text_a",
            "game_b",
            "text_b",
            "blend_type",
            "text_c",
            "score_ac",
            "score_bc",
            "score",
            "sim_ac",
            "sim_bc",
            "sim_diff",
        ]
    ]

    # --------------------------------------------
    # Save CSV
    # --------------------------------------------
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

def eval(config):
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
        trainset_game=getattr(config, "trainset_game", None)
    )

    clip_model = CLIPModel(
        num_classes=val_loader.dataset.num_classes,
        embedding_dim=config.embedding_dim,
        drop_rate=config.drop_rate,
        init_temperature=config.init_temperature,
        text_encoder_model=config.clip_model
    ).to(config.device)

    vae_model = EMA_VQVAE(
        n_channel=val_loader.dataset.num_classes,
        num_codes=config.num_codes,
        code_dim=config.code_dim,
        beta_vq=config.vq_beta,
        cond_dim=config.embedding_dim,
    ).to(config.device)

    # Initialize tokenizer for text processing
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model)

    logger.info("Number of classes: {}".format(val_loader.dataset.num_classes))

    # instantiate checkpoint manager (always create for loading, even if not saving)
    checkpoint_manager = CheckpointManager(
        log_dir_root=config.checkpoint_path,
        config=config,
        save_interval=config.save_interval if config.save_ckpt else 0,
        save_keep=config.save_keep if config.save_ckpt else 0,
        logger=logger,
        get_log_dir_fn=get_log_dir,
    )

    # register models to the checkpoint manager
    checkpoint_manager.register(clip=clip_model, vae=vae_model)

    # Load checkpoint with fallback logic (checkpoint_path > checkpoint_epoch > latest)
    load_checkpoint_with_fallback(
        checkpoint_manager=checkpoint_manager,
        config=config,
        device=config.device,
        model_names=['clip', 'vae'],
        logger=logger
    )

    vae_model.eval()
    clip_model.eval()

    output_dir = os.path.join(
        os.path.dirname(config.exp_path)
        if config.exp_path.endswith(".pt")
        else config.exp_path,
    )
    os.makedirs(output_dir, exist_ok=True)

    output_csv_path = os.path.join(
        output_dir,
        "textblended_instruction.csv"
    )

    vit_evaluator = ViTEvaluator(batch_size=config.vit_batch_size, device=config.device)

    df_blended = pd.read_csv(config.blended_instuction_csv)

    run_textblend_evaluation(
        data_loader=val_loader,
        clip_model=clip_model,
        vae_model=vae_model,
        vit_evaluator=vit_evaluator,
        tokenizer=tokenizer,
        device=config.device,
        df_blended=df_blended,
        output_csv_path=output_csv_path,
    )
    
@hydra.main(version_base="1.3", config_path="conf", config_name="eval_textblend")
def main(config: TextBlendEvalConfig):
    init_config(config)

    logger.info(f"Experiment Directory: {config.exp_path}")

    eval(config)


if __name__ == "__main__":
    main()