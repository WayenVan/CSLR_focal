import os

if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.getcwd(), "src"))
    from csi_sign_language.modules.vit_encoder_return_attn.vision_transformer import (
        VisionTransformer,
    )
else:
    from .vision_transformer import VisionTransformer

from mmpose.apis.inference import init_model
from mmengine import Config

# from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from torch import nn
import torch
from einops import rearrange, reduce
from collections import OrderedDict

from csi_sign_language.utils.data import mapping_0_1
from collections import namedtuple
import re


class VitWithAttnVitPose(nn.Module):
    def __init__(
        self,
        img_size,
        color_range,
        cfg_path,
        checkpoint,
        drop_path_rate,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        cfg = Config.fromfile(cfg_path)

        self.cfg = cfg
        self.color_range = color_range

        self.register_buffer("std", torch.tensor(cfg.model.data_preprocessor.std))
        self.register_buffer("mean", torch.tensor(cfg.model.data_preprocessor.mean))

        # vitpose = init_model(cfg, checkpoint, device="cpu")
        init_args = cfg.model.backbone
        del init_args["type"]
        self.vit = VisionTransformer(**init_args)
        self.migrate_checkpoint(checkpoint)

    def _data_preprocess(self, x):
        x = mapping_0_1(self.color_range, x)
        x = x * 255.0  # mapping to 0-255
        x = x.permute(0, 2, 3, 1)
        x = (x - self.mean) / self.std
        x = x.permute(0, 3, 1, 2)
        return x

    ViTWithAttnVitPoseOut = namedtuple(
        "ViTWithAttnVitPoseOut", ["out", "t_length", "attn_weights"]
    )

    def forward(self, x, t_length):
        N, C, T, H, W = x.shape
        x = rearrange(x, "n c t h w -> (n t) c h w")

        feats, attn_weights = self.vit(x)
        # heatmap_out: list of [n 2 h w] -> [n s 2 h w]
        attn_weights = torch.stack(attn_weights, dim=1)

        attn_weights = rearrange(
            attn_weights, "(n t) s heads keys h w -> t n s heads keys h w", t=T
        )
        out = reduce(feats[-1], "n c h w -> n c", "mean")
        out = rearrange(out, "(n t) c -> n c t", t=T)

        return self.ViTWithAttnVitPoseOut(out, t_length, attn_weights)

    def migrate_checkpoint(self, ckpt):
        checkpoint = torch.load(ckpt, map_location="cpu")
        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if key.startswith("backbone."):
                new_key = re.sub("^backbone\.", "", key)
                new_state_dict[new_key] = value
        self.vit.load_state_dict(new_state_dict, strict=False)
        del checkpoint
        pass

    def train(self, mode: bool = True):
        super().train(mode)

        # for p in self.vit.parameters():
        #     p.requires_grad = not self.freeze_vitpose
        # for p in self.vit_head.parameters():
        #     p.requires_grad = not self.freeze_vitpose

        # if self.freeze_vitpose:
        #     self.vit.eval()
        #     self.vit_head.eval()


if __name__ == "__main__":
    model = VitWithAttnVitPose(
        [224, 224],
        [0, 1],
        cfg_path="resources/ViTPose/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py",
        checkpoint="resources/ViTPose/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth",
        drop_path_rate=0.1,
    ).to("cuda:1")
    x = torch.randn(1, 3, 300, 224, 224).to("cuda:1")
    with torch.no_grad():
        out = model(x, None)
    print(out[0].shape, out[1], out[2].shape)
