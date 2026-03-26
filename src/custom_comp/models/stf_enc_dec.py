from custom_comp.layers import BasicLayer, PatchEmbed, PatchMerging, PatchSplit
from compressai.models.base import CompressionModel

import torch.nn as nn
import torch

class SymmetricalTransFormerEncoder(nn.Module):
    def __init__(self,
                 # ---- only for SymmetricalTransFormer model
                 patch_norm=True,
                 depths=[2, 2, 6, 2],
                 drop_rate=0.,
                 patch_size=2,
                 in_chans=3,
                 num_heads=[3, 6, 12, 24],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 # ---- common w/ Hyp
                 embed_dim=48
                 ):
        print('Creating STF Encoder model')
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = len(depths)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=False)
            self.layers.append(layer)



    def forward(self, x):
        """Forward function."""
        B = x.size(0)
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x, Wh, Ww = layer(x, Wh, Ww)

        y = x
        C = self.embed_dim * 8
        y = y.view(B, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()

        return y




class SymmetricalTransFormerDecoder(nn.Module):
    def __init__(self,
                 # ---- only for SymmetricalTransFormer model
                 patch_norm=True,
                 depths=[2, 2, 6, 2],
                 drop_rate=0.,
                 patch_size=2,
                 in_chans=3,
                 num_heads=[3, 6, 12, 24],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 # ---- common w/ Hyp
                 embed_dim=48
                 ):
        print('Creating STF Decoder model')
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = len(depths)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.syn_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (3-i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchSplit if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=True)
            self.syn_layers.append(layer)

        self.end_conv = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * patch_size ** 2, kernel_size=5, stride=1, padding=2),
                                      nn.PixelShuffle(patch_size),
                                      nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
                                      )
        

    def forward(self, y):

        B, C, Wh, Ww = y.shape
        y_hat = y
        y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(B, Wh*Ww, C)
        for i in range(self.num_layers):
            layer = self.syn_layers[i]
            y_hat, Wh, Ww = layer(y_hat, Wh, Ww)

        x_hat = self.end_conv(y_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous())

        return x_hat


if __name__ == '__main__':

    device = 'cuda'
    encoder = SymmetricalTransFormerEncoder().to(device).eval()
    decoder = SymmetricalTransFormerDecoder().to(device).eval()


    x = torch.rand((1,3,256,256)).to(device)

    y = encoder(x)

    x_hat = decoder(y)

    print(x_hat.shape)