from transformers import ViTForImageClassification, ViTModel
from transformers.models.vit.modeling_vit import ViTAttention, ViTSelfAttention
import torch
from torchvision import transforms
from einops import rearrange, reduce
from collections import namedtuple


class ViTEncoderWithAttentionHG(torch.nn.Module):
    def __init__(self, vit_model, use_cls_token=False):
        super(ViTEncoderWithAttentionHG, self).__init__()
        self.normalizer = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        self.vit_model = ViTModel.from_pretrained(vit_model)
        self.attention_cache = []
        self.attention_hook_handles = []
        self.use_cls_token = use_cls_token

    def pre_forward_hoot(self, module, input):
        # NOTE: this hook is used to set the parameter attention_output to True
        assert isinstance(module, ViTSelfAttention)
        # hidden_states, head_mask, output_attentions
        return (input[0], input[1], True)

    def attention_hook(self, module, input, output):
        assert isinstance(module, ViTSelfAttention)
        attn_weights = output[1]
        # bt, num_heads, t, t NOTE: remove cls token only for keys
        attn_weights = attn_weights[:, :, :, 1:]
        attn_weights = rearrange(
            attn_weights, "b head q (h w) -> b head q h w", h=14, w=14
        )
        # bt, num_heads hw, h, w
        self.attention_cache.append(attn_weights)
        return

    ViTSelfAttentionHGOutput = namedtuple(
        "ViTEncoderWithAttentionHGOutput", ["out", "t_length", "attn_weights"]
    )

    def forward(self, x, t_length):
        # NOTE: make sure catch is clear
        self.attention_cache.clear()
        attn_weight = None
        try:
            if len(self.attention_hook_handles) == 0:
                for layer in self.vit_model.encoder.layer:
                    vit_self_attention = layer.attention.attention
                    pre_hook_handle = vit_self_attention.register_forward_pre_hook(
                        self.pre_forward_hoot
                    )
                    handle = vit_self_attention.register_forward_hook(
                        self.attention_hook
                    )
                    self.attention_hook_handles.append((pre_hook_handle, handle))

            N, C, T, H, W = x.shape
            x = rearrange(x, "n c t h w -> (n t) c h w")

            out = self.vit_model(x).last_hidden_state

            if self.use_cls_token:
                out = out[:, 0, :]
                out = rearrange(out, "(n t) c  -> n c t", n=N)
            else:
                out = out[:, 1:, :]
                out = rearrange(out, "(n t) l c ->  n c l t", t=T)
                out = reduce(out, "n c l t -> n c t", "mean")

            if len(self.attention_hook_handles) > 0:
                attn_weight = torch.stack(self.attention_cache, dim=0)
                attn_weight = rearrange(
                    attn_weight,
                    "s (n t) heads querys h w -> t n s heads querys h w",
                    n=N,
                )

        finally:
            # NOTE: make sure catch is clear
            self.attention_cache.clear()
        return self.ViTSelfAttentionHGOutput(out, t_length, attn_weight)


if __name__ == "__main__":
    model = ViTEncoderWithAttentionHG(vit_model="google/vit-base-patch16-224").cuda()
    model.train()
    x = torch.randn(1, 3, 10, 224, 224).cuda()
    t_length = torch.tensor([10]).cuda()
    out = model(x, t_length)
    for i in out:
        print(i.shape)
