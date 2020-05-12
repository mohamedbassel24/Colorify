import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys

sys.path.append("src")
from resent import resnet101
from skimage.transform import resize
from skimage import io, img_as_float
from skimage.color import rgb2lab, lab2rgb
from tqdm import tqdm
from torchvision.models import mobilenet_v2


class NewSepConv(nn.Module):
    def __init__(self):
        super(NewSepConv, self).__init__()

    def forward(self, imgs, vers, hors):
        b, c, H, W = imgs.size()
        _, s, _, _ = vers.size()

        all_kernels = vers.permute(0, 2, 3, 1).contiguous().view(b, H, W, s, 1) @ hors.permute(0, 2, 3,
                                                                                               1).contiguous().view(b,
                                                                                                                    H,
                                                                                                                    W,
                                                                                                                    1,
                                                                                                                    s)
        all_kernels = all_kernels.view(b, 1, H, W, s, s)
        imgs = torch.nn.ReplicationPad2d([8, 8, 8, 8])(imgs)
        all_patches = []
        for i in range(H):
            for j in range(W):
                all_patches.append(imgs[:, :, i:i + 17, j:j + 17].contiguous().view(b, c, 17, 17, 1))
        all_patches = torch.cat(all_patches, dim=-1).view(b, c, 17, 17, H, W).permute(0, 1, 4, 5, 2, 3).contiguous()
        # print (124,all_patches.size(), all_kernels.size())
        return (all_patches * all_kernels).sum(dim=-1).sum(dim=-1)


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_layers(x)


## LOCAL TRANSFER NET
class LocalTransferNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_32 = TripleConv(6, 32)  # /1
        self.conv_down_32_64 = nn.Sequential(nn.AvgPool2d(2), TripleConv(32, 64))  # /2
        self.conv_down_64_128 = nn.Sequential(nn.AvgPool2d(2), TripleConv(64, 128))  # /4
        self.conv_down_128_256 = nn.Sequential(nn.AvgPool2d(2), TripleConv(128, 256))  # /8
        self.conv_down_25_512 = nn.Sequential(nn.AvgPool2d(2), TripleConv(256, 512))  # /16
        self.conv_down_512_512 = nn.Sequential(nn.AvgPool2d(2), TripleConv(512, 512))  # /32

        self.upsample = nn.Upsample(scale_factor=2)  # /16
        self.conv_up_512_256 = nn.Sequential(TripleConv(512, 256), nn.Upsample(scale_factor=2))  # /8
        self.conv_up_256_128 = nn.Sequential(TripleConv(256, 128), nn.Upsample(scale_factor=2))  # /4
        self.conv_up_128_64 = nn.Sequential(TripleConv(128, 64), nn.Upsample(scale_factor=2))  # /2

        self.image_h_filter = nn.Sequential(TripleConv(64, 17), nn.Upsample(scale_factor=2))  # /1
        self.image_w_filter = nn.Sequential(TripleConv(64, 17), nn.Upsample(scale_factor=2))  # /1
        #         self.image2_h_filter = nn.Sequential(TripleConv(64, 17), nn.Upsample(scale_factor=2)) #/1
        #         self.image2_w_filter = nn.Sequential(TripleConv(64, 17), nn.Upsample(scale_factor=2)) #/1

        self.image_sepconv = NewSepConv()
        # self.image2_sepconv = NewSepConv()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, G_prev, G_cur, I_prev):
        x = self.conv_32(torch.cat((G_prev, G_cur), dim=1))

        x_down_32_64 = self.conv_down_32_64(x)  # /2
        x_down_64_128 = self.conv_down_64_128(x_down_32_64)  # /4
        x_down_128_256 = self.conv_down_128_256(x_down_64_128)  # /8
        x_down_256_512 = self.conv_down_256_512(x_down_128_256)  # /16
        x_down_512_512 = self.conv_down_512_512(x_down_256_512)  # /32

        x_bottle = self.upsample(x_down_512_512)  # /16

        x_up = x_bottle + x_down_256_512
        x_up_512_256 = self.conv_up_512_256(x_up) + x_down_128_256  # /8
        x_up_256_128 = self.conv_up_256_128(x_up_512_256) + x_down_64_128  # /4
        x_up_128_64 = self.conv_up_128_64(x_up_256_128) + x_down_32_64  # /2

        image_kh = self.softmax(self.image_h_filter(x_up_128_64))
        image_kw = self.softmax(self.image_w_filter(x_up_128_64))

        # image2_kh = self.image2_h_filter(x_up_128_64)
        # image2_kw = self.image2_w_filter(x_up_128_64)

        image_sepconv_proceed = self.image_sepconv(I_prev, image_kh, image_kw)
        # image2_sepconv_proceed = self.image2_sepconv(I_prev, image2_kh, image2_kw)

        return image_sepconv_proceed


def infer_batch(batch, refinement_net, local_transfer_net, global_transferer,
                use_only_local=False, use_optical_flow=False, optical_flow_net=None,
                zero_global=False, verbose=False):
    I1_lab = rgb2lab_torch(batch[0].cuda(), use_gpu=True)
    Ik_lab = rgb2lab_torch(batch[2].cuda(), use_gpu=True)

    G1_array = np.array([rgb2gray(img.cpu().numpy()) for img in batch[0]])
    Gk_1_array = np.array([rgb2gray(img.cpu().numpy()) for img in batch[1]])
    Gk_array = np.array([rgb2gray(img.cpu().numpy()) for img in batch[2]])
    Ik_1_array = np.array([img.cpu().numpy() for img in batch[1]])

    G1_tensor = torch.tensor(G1_array.transpose(0, 3, 1, 2), dtype=torch.double, requires_grad=False).cuda()
    Gk_1_tensor = torch.tensor(Gk_1_array.transpose(0, 3, 1, 2), dtype=torch.double, requires_grad=False).cuda()
    Gk_tensor = torch.tensor(Gk_array.transpose(0, 3, 1, 2), dtype=torch.double, requires_grad=False).cuda()

    t = time()
    local_batch_output = local_transfer_net.forward(Gk_1_tensor, Gk_tensor, batch[1].permute(0, 3, 1, 2).cuda())
    if verbose:
        print("Local inference time %.3f" % (time() - t))
    local_batch_output_lab = rgb2lab_torch(local_batch_output.permute(0, 2, 3, 1), use_gpu=True).permute(0, 3, 1, 2)

    t = time()
    global_batch_output = global_transferer.forward(G1_array, Gk_array, batch[0].cpu().numpy())
    if zero_global:
        global_batch_output *= 0
    if verbose:
        print("Global inference time %.3f" % (time() - t))
    global_batch_output = torch.tensor(global_batch_output.transpose(0, 3, 1, 2), dtype=torch.double,
                                       requires_grad=False)
    global_batch_output_lab = rgb2lab_torch(global_batch_output.permute(0, 2, 3, 1), use_gpu=False).permute(0, 3, 1, 2)

    if use_optical_flow:
        optical_flow_result = np.array([get_optical_flow_result(optical_flow_net, Gk_1_array[i],
                                                                Gk_array[i], Ik_1_array[i]) for i in
                                        range(len(Gk_1_array))])
        optical_flow_result = torch.tensor(optical_flow_result.transpose(0, 3, 1, 2), dtype=torch.double,
                                           requires_grad=False)
        optical_flow_result = rgb2lab_torch(optical_flow_result.permute(0, 2, 3, 1), use_gpu=False).permute(0, 3, 1, 2)

        stacked_input_refinement = torch.cat([Gk_tensor, local_batch_output_lab,
                                              global_batch_output_lab.cuda(), optical_flow_result.cuda()], dim=1)
    else:
        optical_flow_result = None
        stacked_input_refinement = torch.cat([Gk_tensor, local_batch_output_lab,
                                              global_batch_output_lab.cuda()], dim=1)
    if use_only_local:
        refinement_output_lab = local_batch_output_lab.permute(0, 2, 3, 1)
        gt_l = Ik_lab[..., 0][..., None]
        predicted_ab = refinement_output_lab[..., -2:]
        result_lab = torch.cat((gt_l, predicted_ab), dim=3)
        result_rgb = lab2rgb_torch(result_lab, use_gpu=True).cpu()
    else:
        refinement_output_lab = refinement_net(stacked_input_refinement).permute(0, 2, 3, 1)
        gt_l = Ik_lab[..., 0][..., None]
        predicted_ab = refinement_output_lab[..., -2:]
        result_lab = torch.cat((gt_l, predicted_ab), dim=3)
        result_rgb = lab2rgb_torch(result_lab, use_gpu=True).cpu()

    if optical_flow_result is None:
        return (result_lab, result_rgb,
                local_batch_output.permute(0, 2, 3, 1), global_batch_output.permute(0, 2, 3, 1))
    else:
        return (result_lab, result_rgb,
                local_batch_output.permute(0, 2, 3, 1), global_batch_output.permute(0, 2, 3, 1),
                optical_flow_result.permute(0, 2, 3, 1))


def frame_to_tensor(frame):
    return torch.tensor(frame, dtype=torch.double, requires_grad=False).cuda()


def inference_test_video(frames, refinement_net, local_transfer_net, global_transferer,
                         use_only_local=False, use_optical_flow=False, optical_flow_net=None, zero_global=False):
    I0 = frames[0]
    I_prev = frames[0]
    Gk_1 = rgb2gray(I_prev)
    G0 = Gk_1.copy()
    output_rgb_frames = [I0]
    output_local = [I0]
    output_global = [I0]
    gray_frames = [G0]
    output_optical = [I0]
    for cur_frame in tqdm(frames[1:]):
        gray_frames.append(rgb2gray(cur_frame))
        batch = (frame_to_tensor(I0[None, ...]),
                 frame_to_tensor(I_prev[None, ...]),
                 frame_to_tensor(cur_frame[None, ...]))

        if not (use_optical_flow):
            _, result_rgb, result_local, result_global = infer_batch(batch, refinement_net, local_transfer_net,
                                                                     global_transferer, use_only_local,
                                                                     use_optical_flow=use_optical_flow,
                                                                     optical_flow_net=optical_flow_net,
                                                                     zero_global=zero_global)
        else:
            _, result_rgb, result_local, result_global, result_optical = infer_batch(batch, refinement_net,
                                                                                     local_transfer_net,
                                                                                     global_transferer, use_only_local,
                                                                                     use_optical_flow=use_optical_flow,
                                                                                     optical_flow_net=optical_flow_net,
                                                                                     zero_global=zero_global)
        output_rgb_frames.append(result_rgb[0].detach().cpu().numpy())
        output_local.append(result_local[0].detach().cpu().numpy())
        output_global.append(result_global[0].detach().cpu().numpy())
        if not (result_optical is None):
            output_optical.append(result_optical)
        I_prev = output_rgb_frames[-1]

    if len(output_optical) == 1:
        return gray_frames, output_rgb_frames, output_local, output_global
    else:
        return gray_frames, output_rgb_frames, output_local, output_global, output_optical
