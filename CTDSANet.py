import torch
import torch.nn as nn
from MACTN.utils import MODEL_REGISTOR, MODEL_REGISTOR_MT, timer_wrap
from MACTN import *
from modules import *
from layers import *
from CMT.transformer import TransformerEncoder

def get_model(model_name, input_shape, output_shape, *args, **kwargs):
    if model_name in MODEL_REGISTOR.registered_names():
        return MODEL_REGISTOR.get(model_name)(input_shape, output_shape, *args, **kwargs)
    else:
        raise NotImplementedError


def get_model_MT(model_name, *args, **kwargs):
    if model_name in MODEL_REGISTOR_MT.registered_names():
        return MODEL_REGISTOR_MT.get(model_name)(*args, **kwargs)
    else:
        raise NotImplementedError


class BaseModel(nn.Module):
    def __init__(self, input_shape=None, output_shape=None):
        super(BaseModel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __build_pseudo_input(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        temp_x_ = torch.rand(input_shape)
        temp_x = temp_x_.unsqueeze(0)
        return temp_x

    def get_tensor_shape(self, forward_func, input_shape=None):
        pseudo_x = self.__build_pseudo_input(input_shape)
        pseudo_y = forward_func(pseudo_x)
        return pseudo_y.shape


@MODEL_REGISTOR.register()
class CL_two_sclaconv_negativeCL(BaseModel):
########### 没有pooling
    def __init__(self, input_shape=(16, 128), output_shape: int = 2,
                 dropoutRate: float = 0.5, kernLength_1: int = 3, kernLength_2: int = 7, num_class: int = 2):
        super().__init__()
        EEG_chan = 8
        PPS_chan = 2
        # chans: int = input_shape[0]
        samples: int = input_shape[-1]
        num_class = num_class
        ###################
        EEG_F1 = EEG_chan * 2
        EEG_F2 = EEG_F1 * 2
        ##################
        PPS_F1 = PPS_chan * 2
        PPS_F2 = PPS_F1 * 2
        ##################
        downSample_1, downSample_2 = 4, 5

############# r1 #####################
        self.stage0_1 = nn.Sequential(
            nn.Conv1d(EEG_chan, EEG_F2, kernLength_1, groups=EEG_chan),
        )
        self.stage1_1 = nn.Sequential(
                    nn.Conv1d(EEG_F2, EEG_F2, kernLength_1, groups=EEG_F2),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.stage2_1 = nn.Sequential(
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_s1_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ####################################################################
        self.stage0_2 = nn.Sequential(
            nn.Conv1d(EEG_chan, EEG_F2, kernLength_2, groups=EEG_chan),
        )
        self.stage1_2 = nn.Sequential(
                    nn.Conv1d(EEG_F2, EEG_F2, kernLength_2, groups=EEG_F2),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.stage2_2 =nn.Sequential(
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_s1_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )


############### r2 ##################

        self.stage0_ECG_1 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_1, groups=PPS_chan),
        )
        self.stage1_ECG_1 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_1, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
        self.stage2_ECG_1 = nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.merge_s1_ECG_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ##############################################################
        self.stage0_ECG_2 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_2, groups=PPS_chan),
        )
        self.stage1_ECG_2 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_2, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.stage2_ECG_2 =nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_s1_ECG_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )

############### r3 ##################

        self.stage0_r3_1 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_1, groups=PPS_chan),
        )
        self.stage1_r3_1 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_1, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.stage2_r3_1 = nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_s1_r3_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ##############################################################
        self.stage0_r3_2 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_2, groups=PPS_chan),
        )
        self.stage1_r3_2 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_2, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.stage2_r3_2 = nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_s1_r3_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )

############### r4 ##################
        self.stage0_r4_1 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_1, groups=PPS_chan),
        )
        self.stage1_r4_1 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_1, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.stage2_r4_1 = nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.merge_s1_r4_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ##############################################################
        self.stage0_r4_2 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_2, groups=PPS_chan),
        )
        self.stage1_r4_2 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_2, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.stage2_r4_2 =nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_s1_r4_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
#################################################################################################################
        ############# r1 #####################
        self.seg0_1 = nn.Sequential(
            nn.Conv1d(EEG_chan, EEG_F2, kernLength_1, groups=EEG_chan),
        )
        self.seg1_1 = nn.Sequential(
                    nn.Conv1d(EEG_F2, EEG_F2, kernLength_1, groups=EEG_F2),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.seg2_1 = nn.Sequential(
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.merge_seg_s1_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ####################################################################
        self.seg0_2 = nn.Sequential(
            nn.Conv1d(EEG_chan, EEG_F2, kernLength_2, groups=EEG_chan),
        )
        self.seg1_2 = nn.Sequential(
                    nn.Conv1d(EEG_F2, EEG_F2, kernLength_2, groups=EEG_F2),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.seg2_2 = nn.Sequential(
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.merge_seg_s1_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )

        ############### r2 ##################
        self.seg0_ECG_1 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_1, groups=PPS_chan),
        )
        self.seg1_ECG_1 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_1, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.seg2_ECG_1 = nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_seg_s1_ECG_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ##############################################################
        self.seg0_ECG_2 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_2, groups=PPS_chan),
        )
        self.seg1_ECG_2 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_2, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.seg2_ECG_2 = nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_seg_s1_ECG_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )

        ############### r3 ##################
        self.seg0_r3_1 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_1, groups=PPS_chan),
        )
        self.seg1_r3_1 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_1, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.seg2_r3_1 = nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_seg_s1_r3_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ##############################################################
        self.seg0_r3_2 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_2, groups=PPS_chan),
        )
        self.seg1_r3_2 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_2, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.seg2_r3_2 =nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )


        self.merge_seg_s1_r3_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )

        ############### r4 ##################
        self.seg0_r4_1 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_1, groups=PPS_chan),
        )
        self.seg1_r4_1 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_1, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.seg2_r4_1 = nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.merge_seg_s1_r4_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ##############################################################
        self.seg0_r4_2 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_2, groups=PPS_chan),
        )
        self.seg1_r4_2 = nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_2, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
        self.seg2_r4_2 = nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )

        self.merge_seg_s1_r4_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
#################################################################################################################

        ###############################################################################
        self.classifier = nn.Sequential(
            nn.Linear(6480, num_class)
        )

        self.pro_1 = nn.Sequential(
            nn.Linear(29, 25)
        )
        self.pro_2 = nn.Sequential(
            nn.Linear(25, 25)
        )

        self.r1_CMT = TransformerEncoder(embed_dim=108,
                                        num_heads=4,  # 8
                                        layers=4,
                                        attn_dropout=0.1,
                                        relu_dropout=0.1,
                                        res_dropout=0.1,
                                        embed_dropout=0.1,
                                        attn_mask=False)
        self.r2_CMT = TransformerEncoder(embed_dim=108,
                                         num_heads=4,  # 8
                                         layers=4,
                                         attn_dropout=0.1,
                                         relu_dropout=0.1,
                                         res_dropout=0.1,
                                         embed_dropout=0.1,
                                         attn_mask=False)
        self.r3_CMT = TransformerEncoder(embed_dim=108,
                                         num_heads=4,  # 8
                                         layers=4,
                                         attn_dropout=0.1,
                                         relu_dropout=0.1,
                                         res_dropout=0.1,
                                         embed_dropout=0.1,
                                         attn_mask=False)
        self.r4_CMT = TransformerEncoder(embed_dim=108,
                                         num_heads=4,  # 8
                                         layers=4,
                                         attn_dropout=0.1,
                                         relu_dropout=0.1,
                                         res_dropout=0.1,
                                         embed_dropout=0.1,
                                         attn_mask=False)

        self.glo_CMT = TransformerEncoder(embed_dim=56,
                                         num_heads=4,  # 8
                                         layers=4,
                                         attn_dropout=0.1,
                                         relu_dropout=0.1,
                                         res_dropout=0.1,
                                         embed_dropout=0.1,
                                         attn_mask=False)
    # @timer_wrap
    def forward(self, x: torch.Tensor):
        x = torch.squeeze(x, dim=1)
        # print(x.shape)
        r1 = x[:, 0:8, :]
        r2 = x[:, 8:10, :]
        r3 = x[:, 10:12, :]
        r4 = x[:, 12:14, :]


        # emg = x[:, 34:36, :]

        r1_embed_private_scale1, r1_embed_private_scale2 = self.forward_embed_r1(r1)
        r1_embed_public_scale1, r1_embed_public_scale2 = self.forward_embed_r1_seg(r1)

        r2_embed_private_scale1, r2_embed_private_scale2 = self.forward_embed_r2(r2)
        r2_embed_public_scale1, r2_embed_public_scale2 = self.forward_embed_r2_seg(r2)

        r3_embed_private_scale1, r3_embed_private_scale2 = self.forward_embed_r3(r3)
        r3_embed_public_scale1, r3_embed_public_scale2 = self.forward_embed_r3_seg(r3)

        r4_embed_private_scale1, r4_embed_private_scale2 = self.forward_embed_r4(r4)
        r4_embed_public_scale1, r4_embed_public_scale2 = self.forward_embed_r4_seg(r4)

        embed_public_scale1 = torch.cat((r1_embed_public_scale1, r2_embed_public_scale1, r3_embed_public_scale1, r4_embed_public_scale1), dim=1)
        embed_public_scale2 = torch.cat((r1_embed_public_scale2, r2_embed_public_scale2, r3_embed_public_scale2, r4_embed_public_scale2), dim=1)

        embed_private_scale1 = torch.cat((r1_embed_private_scale1, r2_embed_private_scale1, r3_embed_private_scale1, r4_embed_private_scale1), dim=1)
        embed_private_scale2 = torch.cat((r1_embed_private_scale2, r2_embed_private_scale2, r3_embed_private_scale2, r4_embed_private_scale2), dim=1)


        pro_1 = self.pro_1(torch.mean(embed_public_scale1, dim=1))
        pro_2 = self.pro_2(torch.mean(embed_public_scale2, dim=1))

        pro_1_private = self.pro_1(torch.mean(embed_private_scale1, dim=1))
        pro_2_private = self.pro_2(torch.mean(embed_private_scale2, dim=1))

        r1_embed = torch.cat(
            (r1_embed_private_scale1, r1_embed_private_scale2, r1_embed_public_scale1, r1_embed_public_scale2), dim=-1)
        r2_embed = torch.cat(
            (r2_embed_private_scale1, r2_embed_private_scale2, r2_embed_public_scale1, r2_embed_public_scale2), dim=-1)
        r3_embed = torch.cat(
            (r3_embed_private_scale1, r3_embed_private_scale2, r3_embed_public_scale1, r3_embed_public_scale2), dim=-1)
        r4_embed = torch.cat(
            (r4_embed_private_scale1, r4_embed_private_scale2, r4_embed_public_scale1, r4_embed_public_scale2), dim=-1)



        region_1 = torch.unsqueeze(torch.mean(r1_embed, dim=1), dim=1)
        region_2 = torch.unsqueeze(torch.mean(r2_embed, dim=1), dim=1)
        region_3 = torch.unsqueeze(torch.mean(r3_embed, dim=1), dim=1)
        region_4 = torch.unsqueeze(torch.mean(r4_embed, dim=1), dim=1)

        Fuse_r1 = torch.cat((region_2, region_3, region_4), dim=1)
        Fuse_r2 = torch.cat((region_1, region_3, region_4), dim=1)
        Fuse_r3 = torch.cat((region_1, region_2, region_4), dim=1)
        Fuse_r4 = torch.cat((region_1, region_2, region_3), dim=1)

        ############ Optimize r1 ################
        r1_fuse = Fuse_r1.permute(1, 0, 2)
        region_1_ = region_1.permute(1, 0, 2)
        R1 = self.r1_CMT(region_1_, r1_fuse, r1_fuse)
        R1_ = R1.permute(1, 0, 2)
        #########################################

        ############ Optimize r2 ################
        r2_fuse = Fuse_r2.permute(1, 0, 2)
        region_2_ = region_2.permute(1, 0, 2)
        R2 = self.r1_CMT(region_2_, r2_fuse, r2_fuse)
        R2_ = R2.permute(1, 0, 2)
        #########################################

        ############ Optimize r3 ################
        r3_fuse = Fuse_r3.permute(1, 0, 2)
        region_3_ = region_3.permute(1, 0, 2)
        R3 = self.r1_CMT(region_3_, r3_fuse, r3_fuse)
        R3_ = R3.permute(1, 0, 2)
        #########################################

        ############ Optimize r3 ################
        r4_fuse = Fuse_r4.permute(1, 0, 2)
        region_4_ = region_4.permute(1, 0, 2)
        R4 = self.r1_CMT(region_4_, r4_fuse, r4_fuse)
        R4_ = R4.permute(1, 0, 2)
        #########################################

        out = torch.cat((R1_, R2_, R3_, R4_), dim=1)
        glo_embed = torch.cat((r1_embed, r2_embed, r3_embed, r4_embed), dim=1)

        glo_embed_ = glo_embed.permute(2, 0, 1)
        glo_ = self.glo_CMT(glo_embed_).permute(1, 2, 0)

        out = torch.concat((out, glo_), dim=1)
        batch = out.shape[0]
        out = out.reshape(batch, -1)
        y2 = self.classifier(out)

        global_embed = torch.squeeze(torch.mean(glo_, dim=1), dim=1)
        region_1 = torch.squeeze(R1_,dim=1)
        region_2 = torch.squeeze(R2_,dim=1)
        region_3 = torch.squeeze(R3_,dim=1)
        region_4 = torch.squeeze(R4_,dim=1)


        return y2, global_embed, region_1, region_2, region_3, region_4, pro_1, pro_2, pro_1_private, pro_2_private
###############################################
    def forward_embed_r1(self, x):
        x_brach_1 = self.stage0_1(x)
        x_brach_1 = self.stage1_1(x_brach_1)
        x_brach_1 = self.merge_s1_1(x_brach_1)
        x_brach_1 = self.stage2_1(x_brach_1)

        x_brach_2 = self.stage0_2(x)
        x_brach_2 = self.stage1_2(x_brach_2)
        x_brach_2 = self.merge_s1_2(x_brach_2)
        x_brach_2 = self.stage2_2(x_brach_2)

        return x_brach_1, x_brach_2

    def forward_embed_r2(self, x):
        x_brach_1 = self.stage0_ECG_1(x)
        x_brach_1 = self.stage1_ECG_1(x_brach_1)
        x_brach_1 = self.merge_s1_ECG_1(x_brach_1)
        x_brach_1 = self.stage2_ECG_1(x_brach_1)

        x_brach_2 = self.stage0_ECG_2(x)
        x_brach_2 = self.stage1_ECG_2(x_brach_2)
        x_brach_2 = self.merge_s1_ECG_2(x_brach_2)
        x_brach_2 = self.stage2_ECG_2(x_brach_2)

        return x_brach_1, x_brach_2


    def forward_embed_r3(self, x):
        x_brach_1 = self.stage0_r3_1(x)
        x_brach_1 = self.stage1_r3_1(x_brach_1)
        x_brach_1 = self.merge_s1_r3_1(x_brach_1)
        x_brach_1 = self.stage2_r3_1(x_brach_1)

        x_brach_2 = self.stage0_r3_2(x)
        x_brach_2 = self.stage1_r3_2(x_brach_2)
        x_brach_2 = self.merge_s1_r3_2(x_brach_2)
        x_brach_2 = self.stage2_r3_2(x_brach_2)

        return x_brach_1, x_brach_2

    def forward_embed_r4(self, x):
        x_brach_1 = self.stage0_r4_1(x)
        x_brach_1 = self.stage1_r4_1(x_brach_1)
        x_brach_1 = self.merge_s1_r4_1(x_brach_1)
        x_brach_1 = self.stage2_r4_1(x_brach_1)

        x_brach_2 = self.stage0_r4_2(x)
        x_brach_2 = self.stage1_r4_2(x_brach_2)
        x_brach_2 = self.merge_s1_r4_2(x_brach_2)
        x_brach_2 = self.stage2_r4_2(x_brach_2)

        return x_brach_1, x_brach_2

##############################################
###############################################
    def forward_embed_r1_seg(self, x):
        x_brach_1 = self.seg0_1(x)
        x_brach_1 = self.seg1_1(x_brach_1)
        x_brach_1 = self.merge_seg_s1_1(x_brach_1)
        x_brach_1 = self.seg2_1(x_brach_1)

        x_brach_2 = self.seg0_2(x)
        x_brach_2 = self.seg1_2(x_brach_2)
        x_brach_2 = self.merge_seg_s1_2(x_brach_2)
        x_brach_2 = self.seg2_2(x_brach_2)


        return x_brach_1, x_brach_2

    def forward_embed_r2_seg(self, x):
        x_brach_1 = self.seg0_ECG_1(x)
        x_brach_1 = self.seg1_ECG_1(x_brach_1)
        x_brach_1 = self.merge_seg_s1_ECG_1(x_brach_1)
        x_brach_1 = self.seg2_ECG_1(x_brach_1)

        x_brach_2 = self.seg0_ECG_2(x)
        x_brach_2 = self.seg1_ECG_2(x_brach_2)
        x_brach_2 = self.merge_seg_s1_ECG_2(x_brach_2)
        x_brach_2 = self.seg2_ECG_2(x_brach_2)


        return x_brach_1, x_brach_2


    def forward_embed_r3_seg(self, x):
        x_brach_1 = self.seg0_r3_1(x)
        x_brach_1 = self.seg1_r3_1(x_brach_1)
        x_brach_1 = self.merge_seg_s1_r3_1(x_brach_1)
        x_brach_1 = self.seg2_r3_1(x_brach_1)

        x_brach_2 = self.seg0_r3_2(x)
        x_brach_2 = self.seg1_r3_2(x_brach_2)
        x_brach_2 = self.merge_seg_s1_r3_2(x_brach_2)
        x_brach_2 = self.seg2_r3_2(x_brach_2)

        return x_brach_1, x_brach_2

    def forward_embed_r4_seg(self, x):
        x_brach_1 = self.seg0_r4_1(x)
        x_brach_1 = self.seg1_r4_1(x_brach_1)
        x_brach_1 = self.merge_seg_s1_r4_1(x_brach_1)
        x_brach_1 = self.seg2_r4_1(x_brach_1)

        x_brach_2 = self.seg0_r4_2(x)
        x_brach_2 = self.seg1_r4_2(x_brach_2)
        x_brach_2 = self.merge_seg_s1_r4_2(x_brach_2)
        x_brach_2 = self.seg2_r4_2(x_brach_2)

        
        return x_brach_1, x_brach_2
