import torch
import pickle
from torch import nn
from torch import Tensor
from torchvision.transforms import functional as F
from mmpose.apis.inference import init_model, init_default_scope
from matplotlib import pyplot as plt
from mmengine import Config
from mmpose.models.heads import RTMCCHead, SimCCHead
from mmpose.models import TopdownPoseEstimator
from mmpose.apis import Pose2DInferencer, inference_topdown
from mmdet.models.backbones import CSPNeXt
from mmpose.codecs.simcc_label import SimCCLabel
from mmpose.models.losses import KLDiscretLoss
from PIL import Image
import numpy as np
from einops import rearrange, einsum


def central_crop(image, crop_width, crop_height):
    # Get the dimensions of the image
    img_width, img_height = image.size

    # Calculate the coordinates for the central crop
    left = (img_width - crop_width) / 2
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_height) / 2

    # Perform the crop
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def decode(x, y, simcc_split_ratio):
    x_locs = x.argmax(dim=-1)
    y_locs = y.argmax(dim=-1)
    locs = torch.stack([x_locs, y_locs], dim=-1).to(x.dtype)
    locs /= simcc_split_ratio
    return locs


class DWPoseWarpper(nn.Module):
    def __init__(self, cfg: str, ckpt: str):
        """
        @param n_decoder_levels: The number of decoder levels to use. If None, all decoder levels are used.
        """
        super(DWPoseWarpper, self).__init__()
        _cfg = Config.fromfile(cfg)
        self.register_buffer("std", torch.tensor(_cfg.model.data_preprocessor.std))
        self.register_buffer("mean", torch.tensor(_cfg.model.data_preprocessor.mean))
        self.dwpose = init_model(_cfg, ckpt, device="cpu")

        self.input_H = _cfg.codec.input_size[1]
        self.input_W = _cfg.codec.input_size[0]
        self.simcc_split_ratio = _cfg.codec.simcc_split_ratio
        self.freeze()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        @param x: The input tensor of shape (B, C, H, W), should be normalized to [0, 1]
        @return: Logits of x, and y location distribution: (B, 144, H*smic_split_ratio, W*smic_split_ratio) * 2
        """
        H, W = x.shape[-2:]

        assert x.max().item() <= 1.0001, (
            "The input tensor should be normalized to [0, 1]"
        )
        assert x.min().item() >= -0.0001, (
            "The input tensor should be normalized to [0, 1]"
        )
        assert (H, W) == (
            self.input_H,
            self.input_W,
        ), (
            "The input tensor should have the same shape as the model input size. required: {}, got: {}".format(
                (self.input_H, self.input_W), (H, W)
            )
        )

        # transer to [0, 255] for adapt the std and mean
        x = x * 255.0
        x = F.normalize(x, self.mean, self.std)
        return self.dwpose(x, None, "tensor")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    model = DWPoseWarpper(
        cfg="resources/dwpose-l/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py",
        ckpt="resources/dwpose-l/dw-ll_ucoco.pth",
    ).to("cuda:1")

    # prepare data
    image = Image.open("./resources/0.jpg")
    # image = central_crop(image, 224, 224)
    image = F.to_tensor(image)
    image = F.resize(image, [256, 192])
    # image = F.pad(image, [0, 0, 0, 64])
    image = image.unsqueeze(0)
    image = image.to("cuda:1")

    # run model
    x, y = model(image)
    print(x.shape)
    print(y.shape)

    pickle.dump(
        {
            "x": x.squeeze(0).cpu().numpy(),
            "y": y.squeeze(0).cpu().numpy(),
        },
        open("outputs/simcc.pkl", "wb"),
    )

    # plt.axis("off")
    # plt.tight_layout(pad=0)  # pad=0 表示无额外边距
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 调整边界
    # plt.imshow(y[:, 5, 100:108].cpu().numpy())
    # plt.savefig("outputs/simcc_y.pdf")
