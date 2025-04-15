import pickle
import numpy as np
from omegaconf import OmegaConf
import sys
import torch.utils.data

sys.path.append("src")
from hydra.utils import instantiate
import torch
from csi_sign_language.data.datamodule.ph14 import Ph14DataModule
from csi_sign_language.models.slr_model import SLRModel
import os
from datetime import datetime
import click


@click.option(
    "--config",
    "-c",
    default="outputs/train/2025-04-14_14-42-15/config.yaml",
)
@click.option(
    "-ckpt",
    "--checkpoint",
    default="outputs/train/2025-04-14_14-42-15/epoch=96_wer-val=25.80_lr=1.00e-08_loss=0.00.ckpt",
)
@click.option("--ph14_root", default="dataset/phoenix2014-release")
@click.option("--ph14_lmdb_root", default="dataset/preprocessed/ph14_lmdb")
@click.option("--index", default=2)
@click.command()
def main(config, checkpoint, ph14_root, ph14_lmdb_root, index):
    current_time = datetime.now()
    file_name = os.path.basename(__file__)
    save_dir = os.path.join(
        "outputs", file_name[:-3], current_time.strftime("%Y-%m-%d_%H-%M-%S")
    )
    cfg = OmegaConf.load(config)

    dm = Ph14DataModule(
        ph14_root,
        ph14_lmdb_root,
        batch_size=1,
        num_workers=6,
        train_shuffle=True,
        val_transform=instantiate(cfg.transforms.test),
        test_transform=instantiate(cfg.transforms.test),
    )
    loader = dm.test_dataloader()
    data = next(iter(loader))

    model = SLRModel.load_from_checkpoint(
        checkpoint, cfg=cfg, map_location="cpu", ctc_search_type="beam", strict=False
    ).cuda()
    model.set_post_process(dm.get_post_process())
    model.train()
    with torch.no_grad():
        video = data["video"].cuda()
        lgt = data["video_length"].cuda()
        outputs, _ = model(video, lgt)

    weight = outputs[2][2].cpu().numpy()
    # [t n s heads keys h w]
    # pickle.dump(weight, open("outputs/attention_weights.pkl", "wb"))


main()
