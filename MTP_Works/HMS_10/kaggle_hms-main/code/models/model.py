import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
# import metaformer_baselines
# from vit import ViT
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models._manipulate import checkpoint_seq
from .wavenet import Wave_Net
from .squeezeformer.model import Squeezeformer
from .eeg_transformer import EEGTransformer
from .eeg_1d import EEGNet
from .conformer import Conformer
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf

    code from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """

        # (N, T, H) -> (N, T) -> (N, T, 1)
        att_w = nn.functional.softmax(self.W(x).squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x

class VitNet(nn.Module):
    def __init__(self, default_configs, device_id):
        super().__init__()
        self.raw_model_type = default_configs["raw_model"]
        self.model_type = default_configs["backbone"]
        if "vit_base" in default_configs["backbone"]:
            self.spec_model = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=True, in_chans=1)
            self.eeg_model = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=True, in_chans=1)
        elif "vit_large" in default_configs["backbone"]:
            self.spec_model = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=True, in_chans=1)
            self.eeg_model = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=True, in_chans=1)
            print("Vit large: vit_small_patch14_reg4_dinov2.lvd142m vit_small_patch14_reg4_dinov2.lvd142m")
        elif "mix" in default_configs["backbone"]:
            self.spec_model = timm.create_model("convnextv2_nano.fcmae_ft_in22k_in1k_384", num_classes=6, pretrained=True, in_chans=1)
            self.eeg_model = timm.create_model("convnext_small.fb_in22k_ft_in1k_384", num_classes=6, pretrained=True, in_chans=1)
        else:
            self.spec_model = timm.create_model(default_configs["backbone"], num_classes=6, pretrained=True, in_chans=1)
            if default_configs["eeg_spec_type"] == "ver1":
                self.eeg_model = timm.create_model(default_configs["backbone"], num_classes=6, pretrained=True, in_chans=1)
            elif default_configs["eeg_spec_type"] == "ver2":
                self.eeg_model = timm.create_model(default_configs["backbone"], num_classes=6, pretrained=True, in_chans=1, dynamic_img_pad=True, dynamic_img_size=True)
        
        if default_configs["raw_model"] == "vit" and "mix" not in default_configs["backbone"]:
            if "vit_base" in default_configs["backbone"]:
                self.raw_50s_model = timm.create_model(default_configs["backbone"], num_classes=6, pretrained=True, in_chans=1)
                self.raw_10s_model = timm.create_model(default_configs["backbone"], num_classes=6, pretrained=True, in_chans=1)
            else:
                print("Vit large: {} vit_small_patch14_reg4_dinov2.lvd142m".format(default_configs["backbone"]))
                self.raw_50s_model = timm.create_model(default_configs["backbone"], num_classes=6, pretrained=True, in_chans=1)
                self.raw_10s_model = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=True, in_chans=1)
        elif "mix" in default_configs["backbone"]:
            self.raw_50s_model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=True, in_chans=1)
            self.raw_10s_model = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=True, in_chans=1)
        elif default_configs["raw_model"] == "convnext":
            self.raw_50s_model = timm.create_model(default_configs["backbone"], num_classes=6, pretrained=True, in_chans=1)
            self.raw_10s_model = timm.create_model(default_configs["backbone"], num_classes=6, pretrained=True, in_chans=1)
        elif default_configs["raw_model"] == "1dcnn":
            self.raw_model = EEGNet(kernels=[3,5,7,9], in_channels=16, fixed_kernel_size=5)
        elif default_configs["raw_model"] == "wavenet":
            self.raw_model = Wave_Net()
        elif default_configs["raw_model"] == "squeezeformer":
            self.raw_model = Squeezeformer(
                input_dim=16,
            )
            # self.raw_model = Squeezeformer(
        #     input_dim=8,
        # )
        # self.raw_model = EEGTransformer()
        # self.rnn = nn.GRU(input_size=16, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.device_id = device_id
        if "coat_lite" not in default_configs["backbone"]:
            self.spec_model.set_grad_checkpointing()
            self.eeg_model.set_grad_checkpointing()
            if default_configs["raw_model"] == "vit":
                self.raw_50s_model.set_grad_checkpointing()
                self.raw_10s_model.set_grad_checkpointing()
            if default_configs["raw_model"] == "convnext":
                self.raw_50s_model.set_grad_checkpointing()
                self.raw_10s_model.set_grad_checkpointing()

        if "vit" in self.model_type:
            self.spec_model.fc_norm = nn.Identity()
            self.spec_model.head_drop = nn.Identity()
            self.spec_model.head = nn.Identity()

            self.eeg_model.fc_norm = nn.Identity()
            self.eeg_model.head_drop = nn.Identity()
            self.eeg_model.head = nn.Identity()
        elif "convnext" in default_configs["backbone"]:
            self.spec_model.head = nn.Identity()
            self.eeg_model.head = nn.Identity()
        elif "mix" in default_configs["backbone"]:
            self.spec_model.head = nn.Identity()
            self.eeg_model.head = nn.Identity()

        if default_configs["raw_model"] == "vit":
            self.raw_50s_model.fc_norm = nn.Identity()
            self.raw_50s_model.head_drop = nn.Identity()
            self.raw_50s_model.head = nn.Identity()
        
            self.raw_10s_model.fc_norm = nn.Identity()
            self.raw_10s_model.head_drop = nn.Identity()
            self.raw_10s_model.head = nn.Identity()
        if default_configs["raw_model"] == "convnext":
            self.raw_50s_model.head = nn.Identity()
            self.raw_10s_model.head = nn.Identity()

        if "convnext" in default_configs["backbone"] or "mix" in default_configs["backbone"]:
            self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")

        if default_configs["raw_model"] == "squeezeformer":
            # self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True, input_fmt="NCHW")
            self.atten_pooling = SelfAttentionPooling(196)
        
        if default_configs["raw_model"] == "vit":
            if "vit_base" in default_configs["backbone"]:
                self.head1 = nn.Linear(384, 6)
                self.head2 = nn.Linear(384, 6)
                self.head3 = nn.Linear(768, 6)
                self.head4 = nn.Linear(768, 6)
                self.head = nn.Linear(384*2+768*2, 6)
            elif "vit_large" in default_configs["backbone"]:
                self.head1 = nn.Linear(384, 6)
                self.head2 = nn.Linear(384, 6)
                self.head3 = nn.Linear(1024, 6)
                self.head4 = nn.Linear(384, 6)
                self.head = nn.Linear(384*3+1024, 6)
            elif "mix" in default_configs["backbone"]:
                self.head1 = nn.Linear(640, 6)
                self.head2 = nn.Linear(768, 6)
                self.head3 = nn.Linear(768, 6)
                self.head4 = nn.Linear(384, 6)
                self.head = nn.Linear(640 + 768 + 768 + 384, 6)
            else:
                self.head1 = nn.Linear(384, 6)
                self.head2 = nn.Linear(384, 6)
                self.head3 = nn.Linear(384, 6)
                self.head4 = nn.Linear(384, 6)
                self.head = nn.Linear(384*4, 6)
        elif default_configs["raw_model"] == "convnext":     
            self.head = nn.Linear(768*4, 6)   
        elif default_configs["raw_model"] == "1dcnn":
            self.head = nn.Linear(384*2 + 280, 6)
        elif default_configs["raw_model"] == "squeezeformer":
            self.head = nn.Linear(384*2 + 196, 6)
       

    def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
        spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        if "vit" in self.model_type:
            spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
            eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
        elif "mix" in self.model_type:
            spec_feature = self.global_pool(self.spec_model.forward_features(spec_imgs))
            eeg_feature = self.global_pool(self.eeg_model.forward_features(eeg_imgs))
        elif "convnext" in self.model_type:
            spec_feature = self.global_pool(self.spec_model.forward_features(spec_imgs))
            eeg_feature = self.global_pool(self.eeg_model.forward_features(eeg_imgs))
        if self.raw_model_type == "vit":
            raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
            raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]
        if self.raw_model_type == "convnext":
            raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_50s_feature = self.global_pool(self.raw_50s_model.forward_features(raw_50s_imgs))
            raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_10s_feature = self.global_pool(self.raw_10s_model.forward_features(raw_10s_imgs))

        if self.raw_model_type == "1dcnn":
            raw_feature = self.raw_model(raw_imgs)
        if self.raw_model_type == "squeezeformer":
            input_lengths = torch.full((raw_imgs.shape[0], ), 2048).to(self.device_id)
            # input_lengths = torch.full((raw_imgs.shape[0], ), 2000).to(self.device_id)
            # raw_imgs = torch.rand((16, 2000, 16)).to(self.device_id)
            raw_50s_feature = self.raw_model(raw_50s_imgs, input_lengths)
            raw_feature = self.atten_pooling(raw_feature)
        

        feature = torch.cat((spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1)
        # feature = raw_feature
        logits_1 = self.head1(spec_feature)
        logits_2 = self.head2(eeg_feature)
        logits_3 = self.head3(raw_50s_feature)
        logits_4 = self.head4(raw_10s_feature)
        
        logits_kl = self.head(feature) 

        return logits_kl, logits_1, logits_2, logits_3, logits_4

if __name__ == '__main__':
    model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=False, in_chans=1).cuda()
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    x = torch.rand((2, 1, 518, 518)).cuda()
    y = model.forward_features(x)
    # y = y.mean(dim=1)
    print(y.shape)
