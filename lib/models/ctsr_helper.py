# Implementation for CTSR Block.

import torch.nn as nn
import numpy as np

class ImplicitRelation(nn.Module):
    def __init__(self, input_size, unit_type="C", mlp_r=1):
        super(ImplicitRelation, self).__init__()
        
        assert unit_type in ["C", "T"], "Relation Unit Type should be C or T."
        self.unit_type = unit_type
        
        if self.unit_type == "C":
            self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, int(input_size / mlp_r)),
            nn.ReLU(),
            nn.Linear(int(input_size / mlp_r), input_size))
        
        self.activate = nn.Sigmoid()
        
    def forward(self, x):
        x_in = x
        
        x_out = self.pooling(x_in)
        x_out = x_out.view(x_out.size(0), -1)

        x_out = self.mlp(x_out)
        x_out = self.activate(x_out)
        
        if self.unit_type == "C":
            return x_out.view(x.size(0), x.size(1), 1, 1, 1).expand_as(x)
        else:
            return x_out.view(x.size(0), x.size(1), x.size(2), 1, 1).expand_as(x)

class ExplicitRelation(nn.Module):
    def __init__(self, num_filters, conv_size, norm_module=nn.BatchNorm3d, freeze_bn=False):
        super(ExplicitRelation, self).__init__()
        
        self.pooling = nn.AdaptiveAvgPool3d((1, None, None))
        
        pad_size = (conv_size - 1) // 2
        self.conv = nn.Conv3d(num_filters, num_filters, kernel_size=(1, conv_size, conv_size), stride=(1, 1, 1), padding=(0, pad_size, pad_size), bias=False)
        self.bn = norm_module(num_features=num_filters, track_running_stats=(not freeze_bn))
        self.activate = nn.Sigmoid()
        
    def forward(self, x):
        x_in = x
        
        x_out = self.pooling(x_in)
        x_out = self.conv(x_out)
        x_out = self.bn(x_out)

        x_out = self.activate(x_out)
        return x_out.expand_as(x)

class CTSR(nn.Module):
    def __init__(self, input_filters, output_filters, temporal_depth, cotere_config, norm_module=nn.BatchNorm3d, freeze_bn=False):
        super(CTSR, self).__init__()

        self.unit = nn.ModuleDict()

        cotere_type = cotere_config['TYPE']
        self.collaborate_type = cotere_config['COLLABORATE_TYPE']
        c_mlp_r = cotere_config['C_MLP_R']
        t_mlp_r = cotere_config['T_MLP_R']
        s_conv_size = cotere_config['S_CONV_SIZE']

        if "C" in cotere_type:
            self.unit["C"] = ImplicitRelation(output_filters, unit_type="C", mlp_r=c_mlp_r)
        
        if "T" in cotere_type:
            self.unit["T"] = ImplicitRelation(output_filters * temporal_depth, unit_type="T", mlp_r=t_mlp_r)
        
        if "S" in cotere_type:
            self.s_relu = nn.ReLU()
            self.unit["S"] = ExplicitRelation(output_filters, s_conv_size, norm_module=norm_module, freeze_bn=freeze_bn)

    def forward(self, x):
        ctsr_in = x

        unit_keys = list(self.unit.keys())
        unit_out_dict = dict()
        for unit_key in unit_keys:
            unit_in = x
            if unit_key == "S":
                unit_in = self.s_relu(unit_in)
            unit_out_dict[unit_key] = self.unit[unit_key](unit_in)

        unit_out = None
        if self.collaborate_type == "mul_sum":
            if "T" in unit_keys and "S" in unit_keys:
                unit_out = unit_out_dict["T"] + unit_out_dict["S"]
                if "C" in unit_keys:
                    unit_out = unit_out_dict["C"] * unit_out
            else:
                unit_out = unit_out_dict[unit_keys[0]]
                if len(unit_keys) == 2:
                    unit_out = unit_out * unit_out_dict[unit_keys[1]]
        else:
            for i in range(len(unit_keys)):
                if i == 0:
                    unit_out = unit_out_dict[unit_keys[i]]
                else:
                    unit_out = unit_out + unit_out_dict[unit_keys[i]]

        return ctsr_in * unit_out
