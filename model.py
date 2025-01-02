import torch
import torch.nn as nn
import numpy as np
from uniformer import *


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        x = self.sigmoid(x)
        out = residual * x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.softmax(max_out)
        out = out * x

        return out


class Global_net(nn.Module):
    def __init__(self, inchannel):
        super(Global_net, self).__init__()
        self.alpha = nn.Conv1d(inchannel, inchannel, kernel_size=1, padding=0)
        self.beta = nn.Conv1d(inchannel, inchannel, kernel_size=1, padding=0)

        self.d1 = nn.Conv1d(inchannel, inchannel//4, kernel_size=3, padding=1, dilation=1)
        self.d3 = nn.Conv1d(inchannel, inchannel//4, kernel_size=3, padding=3, dilation=3)
        self.d5 = nn.Conv1d(inchannel, inchannel//4, kernel_size=3, padding=5, dilation=5)
        self.d7 = nn.Conv1d(inchannel, inchannel//4, kernel_size=3, padding=7, dilation=7)

        self.fia= nn.Conv1d(inchannel, inchannel, kernel_size=1, padding=0)
        self.gma = nn.Conv1d(inchannel, inchannel, kernel_size=1, padding=0)

        self.conv3_1 = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(inchannel),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(inchannel),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(2 * inchannel, inchannel, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(inchannel),
                                     nn.ReLU())
        self.weight = weight(1)

    def forward(self, r, d):
        weight_r, weight_d = self.weight(r, r, d, d)

        mul_r_d = r.mul(d)
        add_r_d = r + d
        input = mul_r_d + add_r_d
        b, c, h, w = input.size()
        input_re = input.reshape(b, c, -1)

        alpha = self.alpha(input_re) #b, c, hw
        alpha_per = alpha.permute(0, 2, 1) #b, hw, c
        beta = self.beta(input_re) #b, c, hw
        mul_ba = torch.bmm(beta, alpha_per) #b, c, c

        mul_ba_d1 = self.d1(mul_ba)
        mul_ba_d3 = self.d3(mul_ba)
        mul_ba_d5 = self.d5(mul_ba)
        mul_ba_d7 = self.d7(mul_ba)
        cat_all = torch.cat([mul_ba_d1, mul_ba_d3, mul_ba_d5, mul_ba_d7], dim=1)
        cat_all_residual = cat_all + mul_ba

        cat_all_residual_fia = self.fia(cat_all_residual)
        mul_new = torch.bmm(cat_all_residual_fia, alpha)
        mul_new_gma = self.gma(mul_new)
        mul_new_gma_re = mul_new_gma.reshape(b, c, h, w)
        input_new = input + mul_new_gma_re

        r_new1 = (r * weight_r) + input_new
        r_new = self.conv3_1(r_new1)
        d_new1 = (d * weight_d) + input_new
        d_new = self.conv3_2(d_new1)

        out1 = torch.cat((r_new, d_new), dim=1)
        out = self.conv3_3(out1)

        return  out


class purification_module(nn.Module):
    def __init__(self, inchannel):
        super(purification_module, self).__init__()

        self.conv0r = nn.Conv2d(inchannel, inchannel, kernel_size=5, padding=2, groups=inchannel)
        self.conv0d = nn.Conv2d(inchannel, inchannel, kernel_size=5, padding=2, groups=inchannel)

        self.conv_spatial_r = nn.Conv2d(inchannel, inchannel, kernel_size=7, padding=9, groups=inchannel, dilation=3)
        self.conv_spatial_d = nn.Conv2d(inchannel, inchannel, kernel_size=7, padding=9, groups=inchannel, dilation=3)

        self.conv1_r = nn.Conv2d(inchannel, inchannel, kernel_size=1)
        self.conv1_d = nn.Conv2d(inchannel, inchannel, kernel_size=1)

        self.weight = weight(1)


    def forward(self, r, d):
        weight_r, weight_d = self.weight(r, r, d, d)

        r_w = r * weight_r
        d_w = d * weight_d

        add_rd = r_w + d_w
        mul_rd = r_w * d_w

        add_rd_conv0r = self.conv0r(add_rd)
        mul_rd_conv0d = self.conv0d(mul_rd)

        add_inter = add_rd_conv0r + mul_rd_conv0d
        add_inter_conv_spatial_r = self.conv_spatial_r(add_inter)
        add_inter_conv_spatial_d = self.conv_spatial_d(add_inter)

        add_inter_conv_spatial_r_1r = self.conv1_r(add_inter_conv_spatial_r)
        add_inter_conv_spatial_d_1d = self.conv1_d(add_inter_conv_spatial_d)

        r_new = add_inter_conv_spatial_r_1r * add_rd
        d_new = add_inter_conv_spatial_d_1d * mul_rd
        out = r_new + d_new

        return out



class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, scale):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.up(x)

        return out


class weight(nn.Module):
    def __init__(self, n):
        super(weight, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.up8 = nn.Upsample(scale_factor=n, mode='bilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, r1, r4, d1, d4):
        r1_max = self.sigmoid(self.max(r1))
        r1_mul = r1 * r1_max
        r1_mean = torch.mean(r1_mul, dim=1, keepdim=True)

        r4 = self.up8(r4)
        r4_avg = self.sigmoid(self.avg(r4))
        r4_mul = r4 * r4_avg
        r4_mean = torch.mean(r4_mul, dim=1, keepdim=True)
        add_14_r = r1_mean + r4_mean
        add_14_r_relu = self.relu(add_14_r)

        d1_max = self.sigmoid(self.max(d1))
        d1_mul = d1 * d1_max
        d1_mean = torch.mean(d1_mul, dim=1, keepdim=True)

        d4 = self.up8(d4)
        d4_avg = self.sigmoid(self.avg(d4))
        d4_mul = d4 * d4_avg
        d4_mean = torch.mean(d4_mul, dim=1, keepdim=True)
        add_14_d = d1_mean + d4_mean
        add_14_d_relu = self.relu(add_14_d)

        sum = torch.cat((add_14_r_relu, add_14_d_relu), dim=1)
        sum_softmax = self.softmax(sum)
        sum_softmax_r, sum_softmax_d = sum_softmax.split(1, dim=1)

        weight_r = self.avg(sum_softmax_r)

        weight_d = self.avg(sum_softmax_d)

        return weight_r, weight_d



class AMENet(nn.Module):
    def __init__(self, norm_layer = nn.LayerNorm):
        super(AMENet, self).__init__()

        self.rgb_segformer = uniformer_small()
        self.depth_segformer = uniformer_small()

        self.purification_module_f1 = purification_module(64)
        self.purification_module_f2 = purification_module(128)
        self.purification_module_f3 = purification_module(320)
        self.purification_module_f4 = Global_net(512)


        self.conv12_3 = nn.Conv2d(12, 3, 1)

        self.decoder_s1 = Decoder(512, 320, 2)
        self.decoder_s2 = Decoder(320, 128, 2)
        self.decoder_s3 = Decoder(128, 64, 2)
        self.decoder_s4 = Decoder(64, 32, 1)

        self.conv32_s = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        self.conv64_s = nn.Sequential(nn.Conv2d(64, 1, 1),
                                    nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        self.conv128_s = nn.Sequential(nn.Conv2d(128, 1, 1),
                                     nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
        self.conv320_s = nn.Sequential(nn.Conv2d(320, 1, 1),
                                      nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True))

        self.conv64_32_r = nn.Sequential(nn.Conv2d(64, 32, 1))
        self.conv64_32_d = nn.Sequential(nn.Conv2d(64, 32, 1))
        self.conv512_32_r = nn.Sequential(nn.Conv2d(512, 32, 1),
                                         nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
        self.conv512_32_d = nn.Sequential(nn.Conv2d(512, 32, 1),
                                         nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))


    def forward(self,r ,d):
        ddd = torch.cat([d, d, d], dim=1)

        rgb_list = self.rgb_segformer(r)
        depth_list = self.depth_segformer(ddd)

        r1 = rgb_list[0]
        r2 = rgb_list[1]
        r3 = rgb_list[2]
        r4 = rgb_list[3]

        d1 = depth_list[0]
        d2 = depth_list[1]
        d3 = depth_list[2]
        d4 = depth_list[3]


        rd1  = self.purification_module_f1(r1, d1)
        rd2  = self.purification_module_f2(r2, d2)
        rd3 = self.purification_module_f3(r3, d3)
        rd4 = self.purification_module_f4(r4, d4)


        s4 = self.decoder_s1(rd4)
        s31 = s4 + rd3
        s3 = self.decoder_s2(s31)
        s21 = s3 + rd2
        s2 = self.decoder_s3(s21)
        s11 = s2 + rd1
        s1 = self.decoder_s4(s11)

        s12 = self.conv32_s(s1)
        s22 = self.conv64_s(s2)
        s32 = self.conv128_s(s3)
        s42 = self.conv320_s(s4)

        r1_up2 = self.conv64_32_r(r1)
        d1_up2 = self.conv64_32_d(d1)
        b1, c1, h1, w1 = r1_up2.shape
        r1_up2_reshape = r1_up2.reshape(b1, c1, -1)
        r1_up2_tr = r1_up2_reshape.permute(0, 2, 1)
        d1_up2_reshape = d1_up2.reshape(b1, c1, -1)
        d1_up2_tr = d1_up2_reshape.permute(0, 2, 1)
        r1_new = torch.bmm(r1_up2_tr, r1_up2_reshape)
        r1_new = r1_new.reshape(b1, -1, h1*w1, h1*w1)
        d1_new = torch.bmm(d1_up2_tr, d1_up2_reshape)
        d1_new = d1_new.reshape(b1, -1, h1 * w1, h1 * w1)

        r4_up2 = self.conv512_32_r(r4)
        d4_up2 = self.conv512_32_d(d4)
        b2, c2, h2, w2 = r4_up2.shape
        r4_up2_reshape = r4_up2.reshape(b2, c2, -1)
        r4_up2_tr = r4_up2_reshape.permute(0, 2, 1)
        d4_up2_reshape = d4_up2.reshape(b2, c2, -1)
        d4_up2_tr = d4_up2_reshape.permute(0, 2, 1)
        r4_new = torch.bmm(r4_up2_tr, r4_up2_reshape)
        r4_new = r4_new.reshape(b2, -1, h2 * w2, h2 * w2)
        d4_new = torch.bmm(d4_up2_tr, d4_up2_reshape)
        d4_new = d4_new.reshape(b2, -1, h2 * w2, h2 * w2)

        s1_reshape = s1.reshape(b1, c1, -1)
        s1_tr = s1_reshape.permute(0, 2, 1)
        s1_new = torch.bmm(s1_tr, s1_reshape)
        s1_new = s1_new.reshape(b1, -1, h1 * w1, h1 * w1)

        return s12, s22, s32, s42, r1_new, d1_new, r4_new, d4_new, s1_new


    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.rgb_segformer.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.rgb_segformer.load_state_dict(model_dict_r)

        model_dict_d = self.depth_segformer.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        model_dict_d.update(state_dict_d)
        self.depth_segformer.load_state_dict(model_dict_d)
        # self.rgb_depth.load_state_dict(new_state_dict, strict=False)
        print('self.rgb_uniforr loading', 'self.depth_unifor loading')


if __name__ == '__main__':
    pre_path = '/pre_train_path/mask_rcnn_1x_hybrid_base.pth'
    a = np.random.random((4,3,224,224))
    b = np.random.random((4,1,224,224))
    c = torch.Tensor(a).cuda()
    d = torch.Tensor(b).cuda()
    swinNet = AMENet().cuda()
    swinNet.load_pre(pre_path)
    swinNet(c,d)