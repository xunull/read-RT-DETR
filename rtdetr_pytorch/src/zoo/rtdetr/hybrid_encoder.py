'''by lyuwenyu
'''

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from src.core import register

__all__ = ['HybridEncoder']


# 一次conv卷积
# 卷积然后norm然后激活
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# 两层卷积
class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        # 推理的时候
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            # 训练的时候
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    # deploy时，将两层conv的weight和bias进行了融合，置入一个conv中
    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


# Figure 4的那个操作
class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        # expansion=1时没有变化
        # expansion=0.5 卷积时内部的channel数量为一半
        hidden_channels = int(out_channels * expansion)
        # 图4，上下两条线的1x1 conv
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        # 重复N词
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        # 如果设置的expansion != 1的时候，如0.5,那么最后要将channel变回去
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    # Figure 4的操作
    def forward(self, x):
        # Figure 4 的下路
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        # Figure 4 的上路
        x_2 = self.conv2(x)
        # 上下路 element-wise add
        return self.conv3(x_1 + x_2)


# encoder layer，就是一个正常的attention
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


# 被HybridEncoder使用
# 包装上面的encoder，主要是处理几层encoder
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        num_layers 只有1，论文中提到了只使用一层注意力计算
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        # 经过encoder，上面定义的那个TransformerEncoderLayer
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


# encoder
@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 # s3 s4 s5, s5的index为2
                 # 只有s5 进行attention计算
                 use_encoder_idx=[2],
                 # encoder的层数
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 # todo
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # 2d feature to transformer token
        # channel projection
        self.input_proj = nn.ModuleList()

        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        # 几层encoder（论文中是1个）
        # use_encoder_idx 只有最后一层S5，同时对S5仅使用一个encoder层
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()

        # 2 1
        for _ in range(len(in_channels) - 1, 0, -1):
            # 一层卷积 Figure 3 中的黄色块
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            # Figure4 那个操作
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()

        # 0 1
        for _ in range(len(in_channels) - 1):
            # 一层卷积 Figure3 中的蓝色块，下采样，就是继续卷积一下
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        高频位置编码
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # 顺序是 s3,s4,s5
        # [bs,c,h,w]
        # 2d feature to transformer token
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # 论文中使用了1层
        # encoder
        if self.num_encoder_layers > 0:
            # 只有s5 输入到encoder中，s3 s4的特征不进入encoder
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                # 高宽推平，然后和channel换位
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                # 经过encoder,s5的特征经过了encoder
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                # 将s5输出的结果放回到proj_feats
                # [bs,hw,256] -> [bs,256,hw] -> [bs,256,h,w]
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        # s5
        inner_outs = [proj_feats[-1]]

        # idx = 2, 1
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # 第一次循环是S5（经过Attention的）,s5在上面放进去的，当时只有s5
            # 第二次时，是insert 0 的，计算后的结果
            feat_heigh = inner_outs[0]
            # 1, 0
            # 第一次是S4，第二次是S3
            feat_low = proj_feats[idx - 1]
            # 0, 1
            # Figure3 中的黄色块
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            # 更新了inner_outs[0]
            inner_outs[0] = feat_heigh
            # 上采样
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # concat，对应于Figure4中一开始就将两个特征进行cat
            # 取第0 1 两个CCFM
            # S5 经过上采样之后的特征 和 S4的特征进行CCFM
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            # 插入到最前方，上面取feat_heigh的又是一个新的
            # 在一次循环中，更新了两次inner_outs的内容
            inner_outs.insert(0, inner_out)
        # 第一次循环后, inner_outs的内容为 [CCFM(S5->黄色块->上采样，S4), S5->黄色块]
        # 第二次循环使用的是 CCFM(S5->黄色块->上采样，S4)的结果，经过一个黄色块然后上采样 和 S3的内容进行CCFM
        # 第二次循环后, inner_outs的内容为 [CCFM( CCFM(S5->黄色块->上采样，S4)的结果，经过一个黄色块然后上采样, S3),
        #                                CCFM(S5->黄色块->上采样，S4)的结果，经过一个黄色块,
        #                                S5->黄色块]

        # outs 为 CCFM( CCFM(S5->黄色块->上采样，S4)的结果，经过一个黄色块然后上采样, S3)
        # 第一个为 Figure 3中最下面Fusion向上的蓝色线
        outs = [inner_outs[0]]

        # 0 1
        for idx in range(len(self.in_channels) - 1):
            # 第一次为 CCFM( CCFM(S5->黄色块->上采样，S4)的结果，经过一个黄色块然后上采样, S3) 的结果
            # 第二次为上一次的结果
            feat_low = outs[-1]
            # 第一次为 CCFM(S5->黄色块->上采样，S4)的结果，经过一个黄色块
            feat_height = inner_outs[idx + 1]
            # 下采样
            # 第一次为 CCFM( CCFM(S5->黄色块->上采样，S4)的结果，经过一个黄色块然后上采样, S3) 的结果 然后上采样
            downsample_feat = self.downsample_convs[idx](feat_low)
            # 经过CCFM
            # 第一次为 CCFM( CCFM(S5->黄色块->上采样，S4)的结果，经过一个黄色块然后上采样, S3) 的结果 然后上采样
            # 和  第一次为 CCFM(S5->黄色块->上采样，S4)的结果，经过一个黄色块

            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))

            # 第一次添加的为Figure 3中 CCFM中的第二个黑线
            # 第二次添加的为Figure 3中 CCFM中的最上面的黑线
            outs.append(out)

        return outs
