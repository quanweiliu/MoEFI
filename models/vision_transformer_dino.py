# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from typing import Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
# from layers import Mlp, PatchEmbed, CustomSmallPatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
# from modules import ModalitySpecificMoE_ViT

from .layers import Mlp, PatchEmbed, CustomSmallPatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from .modules import ModalitySpecificMoE_ViT

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if npatch == N and w == h:
            return self.pos_embed
        
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        # print("x shape:", x.shape)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # print("cat output shape", x.shape)         # [B, 4, 384]/[B, 1, 384]

        # print("pos_embed shape:", self.pos_embed.shape)
        x = x + self.interpolate_pos_encoding(x, w, h)
        # print("output shape", x.shape)             # [B, 4, 384]/[B, 1, 384]

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        # print("x_norm shape:", x_norm.shape)  # [2, 37, 384]
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            # print("training")
            return ret
        else:
            # print("Not training")
            return self.head(ret["x_prenorm"])
            return self.head(ret["x_norm_clstoken"])

def visualize_patch_token_map(image_np, x_norm_patchtokens, patch_size):
    """
    image_np: 原图 numpy (H, W, 3)
    x_norm_patchtokens: tensor [1, N_patch, C]
    patch_size: patch 的尺寸（比如 16）
    """
    B, N_patch, C = x_norm_patchtokens.shape
    H, W, _ = image_np.shape
    h_feat = H // patch_size
    w_feat = W // patch_size

    # 取每个 patch 的 token 向量 norm
    heatmap = x_norm_patchtokens[0].norm(dim=-1).reshape(h_feat, w_feat)

    # 归一化
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap

    # # 插值到原图大小
    # heatmap = cv2.resize(heatmap.cpu().numpy(), (W, H))

    # # 可视化
    # plt.imshow(image_np)
    # plt.imshow(heatmap, cmap='jet', alpha=0.5)
    # plt.axis('off')
    # plt.title('Patch token activation heatmap')
    # plt.show()

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def load_dinov2_transformer_only(model, pretrained_path):
    state_dict = torch.load(pretrained_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    filtered = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("patch_embed") or k.startswith("pos_embed") or k.startswith("cls_token"))
    }
    msg = model.load_state_dict(filtered, strict=False)
    print("✅ Transformer weights loaded (without patch_embed):", msg)


class Vit_base(nn.Module):
    def __init__(self, channell, channel2, img_size=6, patch_size=2, selected_layers=[0,1,2,3,4,5,6,7,8,9,10,11], **kwargs):
        super(Vit_base, self).__init__()
        
        self.model1 = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=channell,
        embed_dim=384,
        depth=len(selected_layers),
        num_heads=6,
        mlp_ratio=4.0,
        embed_layer=CustomSmallPatchEmbed,
        block_fn=partial(Block, attn_class=MemEffAttention),
        block_chunks=0,
    )
        
        self.model2 = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=channel2,
        embed_dim=384,
        depth=len(selected_layers),
        num_heads=6,
        mlp_ratio=4.0,
        embed_layer=CustomSmallPatchEmbed,
        block_fn=partial(Block, attn_class=MemEffAttention),
        block_chunks=0,
    )
        # adjust along the input image size
        # self.SpMoE = ModalitySpecificMoE_ViT(10)

        load_selected_blocks(self.model1, \
                        "/home/icclab/Documents/lqw/Multimodal_Classification/MoEIF/pretrain/dinov2_vits14_pretrain.pth",
                        selected_layers=selected_layers)
        load_selected_blocks(self.model2, \
                        "/home/icclab/Documents/lqw/Multimodal_Classification/MoEIF/pretrain/dinov2_vits14_pretrain.pth",
                        selected_layers=selected_layers)

    def get_visulization(self, x, y):
        x1 = self.model1(x)
        y1 = self.model2(y)
        # print(x1.shape, y1.shape)  # 2, 37, 384
        center = x1 + y1

        return x1, y1, center
    

    def forward(self, x, y):
        x1 = self.model1(x)
        y1 = self.model2(y)
        # print(x1.shape, y1.shape)  # 2, 37, 384

        center = x1 + y1
        # center = self.SpMoE(x1, y1)   
        # print(x1.shape, y1.shape, center.shape)

        xoutput = x1[:, 0]
        youtput = y1[:, 0]
        output = center[:, 0]

        return xoutput, youtput, output


def vit_hsi(in_chans=3, img_size=6, patch_size=2,  **kwargs):
    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size, 
        in_chans=in_chans,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        embed_layer=CustomSmallPatchEmbed,  # 替换 patch_embed
        block_fn=partial(Block, attn_class=MemEffAttention),
        block_chunks=0,
    )
    return model


# for vit_small
def vit_selected_small(in_chans=3, img_size=6, patch_size=2, selected_layers=[0, 2, 5, 7], **kwargs):
    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=384,
        depth=len(selected_layers),
        num_heads=6,
        mlp_ratio=4.0,
        embed_layer=CustomSmallPatchEmbed,
        block_fn=partial(Block, attn_class=MemEffAttention),
        block_chunks=0,
    )
    return model


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_selected_base(in_chans=3, img_size=6, patch_size=2, selected_layers=[0, 2, 5, 7], **kwargs):
    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=768,
        depth=len(selected_layers),
        num_heads=12,
        mlp_ratio=4.0,
        embed_layer=CustomSmallPatchEmbed,
        block_fn=partial(Block, attn_class=MemEffAttention),
        block_chunks=0,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def load_transformer(model, pretrained_path):
    state_dict = torch.load(pretrained_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    filtered = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("patch_embed") or k.startswith("pos_embed") or k.startswith("cls_token"))
    }
    msg = model.load_state_dict(filtered, strict=False)
    print("✅ Transformer weights loaded (without patch_embed):", msg)


def load_selected_blocks(model, pretrained_path, selected_layers=[0, 2, 5, 7]):
    """
    仅加载指定的 transformer block 层，并重命名权重以适配当前模型。
    参数：
        model: 精简后的 ViT 模型（如 depth=4）
        pretrained_path: DINOv2 的预训练权重路径
        selected_layers: 要加载的 block 层编号列表，例如 [0, 2, 5, 7]
    """
    state_dict = torch.load(pretrained_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    filtered = {}

    layer_mapping = {pre_idx: new_idx for new_idx, pre_idx in enumerate(selected_layers)}
    for k, v in state_dict.items():
        if k.startswith("blocks."):
            parts = k.split(".")
            layer_idx = int(parts[1])
            if layer_idx in selected_layers:
                new_k = k.replace(f"blocks.{layer_idx}", f"blocks.{layer_mapping[layer_idx]}")
                filtered[new_k] = v
        elif not (k.startswith("patch_embed") or k.startswith("pos_embed") or k.startswith("cls_token")):
            # 保留其他非 block 权重（如 norm）
            filtered[k] = v

    msg = model.load_state_dict(filtered, strict=False)
    print(f"✅ Selected transformer blocks loaded: {selected_layers} -> [0, 1, 2, ...]")
    print("   State dict loading report:", msg)


if __name__ == "__main__":
    img_size = 6
    x = torch.randn(2, 15, img_size, img_size).cuda()  # batch size=2
    y = torch.randn(2, 1, img_size, img_size).cuda()  # batch size=2
    model = Vit_base(channell=15, channel2=1, img_size=img_size).cuda()
    output1, outpu2, output3 = model(x, y)
    print("output shape:", output1.shape, outpu2.shape, output3.shape)  # [2, 37, 384] from cls


# if __name__ == "__main__":

#     img_size = 9
#     patch_size = 3
#     x = torch.randn(2, 3, img_size, img_size)  # batch size=2
    
#     model = vit_hsi(img_size=img_size, patch_size=patch_size, in_chans=3)

#     # 3. 加载权重：只加载 transformer 编码器层

#     # 示例调用（请替换你的路径）
#     load_transformer(model, "/home/icclab/Documents/lqw/Multimodal_Classification/KnowCL_competitive/weights/dinov2_vits14_pretrain.pth")

#     # 4. 模拟输入图像（5x5）
#     with torch.no_grad():
#         out = model(x)
#         print("output shape:", out.shape)  # [2, 768] from cls token

# ######################################################################################

#     selected_layers = [0, 2, 5, 7]  # 从 dinov2 中选择的层
#     model = vit_selected_small(in_chans=3, img_size=img_size, patch_size=patch_size, selected_layers=[0, 2, 5, 7])
#     load_selected_blocks(model, "/home/icclab/Documents/lqw/Multimodal_Classification/KnowCLPlus/weights/dinov2_vits14_pretrain.pth", selected_layers)

#     # 4. 模拟输入图像（5x5）
#     with torch.no_grad():
#         out = model(x)
#         print("output shape:", out.shape)  # [2, 768] from cls token