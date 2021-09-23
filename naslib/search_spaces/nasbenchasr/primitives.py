# defines the NASBench-ASR primitives, move to core in future
# Copyright @ NB-ASR authors
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from naslib.search_spaces.core.primitives import AbstractPrimitive


def get_loss():
    def loss(output, output_len, targets, targets_len):
        output_trans = output.permute(1, 0, 2) # needed by the CTCLoss
        loss = F.ctc_loss(output_trans, targets, output_len, targets_len, reduction='none', zero_infinity=True)
        loss /= output_len
        loss = loss.mean()
        return loss

    return loss



class ASRPrimitive(AbstractPrimitive):

    def get_op_name(self):
        if hasattr(self, 'name'):
            return self.name
        return super().get_op_name()
    
    def get_embedded_ops(self):
        return None

    def forward(self, x, edge_data=None):
        raise NotImplementedError()


class PadConvRelu(ASRPrimitive):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, strides, groups=1, dropout_rate=0, context=4, name='PadConvRelu', skip_connection=False):
        super().__init__(locals())
        self.name = name

        if int(context / strides) >= (kernel_size*dilation-strides):
            rpad = kernel_size*dilation-strides
            lpad = 0
        else:
            rpad = int(context / strides)
            lpad = int((kernel_size - 1)*dilation - rpad)

        self.pad = nn.ZeroPad2d((lpad, rpad, 0, 0))
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=strides, dilation=dilation, groups=groups)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.skip_connection = skip_connection

    def forward(self, x, edge_data=None):
        x_in = x
        # import ipdb; ipdb.set_trace()
        x = self.pad(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.clamp_max_(x, 20)
        x = self.dropout(x)
        if self.skip_connection:
            x += x_in

        return x
    
    def __repr__(self):
        return f'{self.name}({self.conv})'

class PadConvReluNorm(PadConvRelu):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm = nn.LayerNorm(kwargs['out_channels'], eps=0.001)

    def forward(self, x, edge_data=None):
        x = super().forward(x, edge_data)
        x = x.permute(0,2,1)
        x = self.norm(x)
        x = x.permute(0,2,1)
        return x

class Linear(ASRPrimitive):
    def __init__(self, in_features, out_features, dropout_rate=0, name='Linear', skip_connection=False):
        super().__init__(locals())
        self.name = name

        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.skip_connection = skip_connection

    def forward(self, x, edge_data=None):
        shape = x.shape
        x_in = x
        x = x.permute(0,2,1)
        x = self.linear(x)
        x = self.relu(x)
        x = torch.clamp_max_(x, 20)
        x = self.dropout(x)
        x = x.permute(0,2,1)
        if self.skip_connection:
            x += x_in
        return x
    
    def __repr__(self):
        return f'{self.__class__}({self.linear})'


class Identity(ASRPrimitive):
    def __init__(self, name='identity'):
        super().__init__(locals())
        self.name = name

    def forward(self, x, edge_data=None):
        return x

class Zero(ASRPrimitive):
    def __init__(self, name='zero', skip_connection=False):
        super().__init__(locals())
        self.name = name
        self.skip_connection = skip_connection

    def forward(self, x, edge_data=None):
        out = torch.zeros_like(x)
        if self.skip_connection:
            out += x
        return out
    def __repr__(self):
        return f'{self.__class__}({"skip" if self.skip_connection else ""})'

class CellLayerNorm(ASRPrimitive):

    def __init__(self, filters, eps=0.001):
        super().__init__(locals())
        self.norm_layer = nn.LayerNorm(filters, eps)
    
    def forward(self, x, edge_data=None):
        output = x.permute(0,2,1)
        output = self.norm_layer(output)
        output = output.permute(0,2,1)
        return x


class Head(ASRPrimitive):

    def __init__(self, config):
        super().__init__(locals())
        self.config = config
        self.layers = nn.ModuleList([
            nn.Dropout(config.dropout_rate),
            nn.LSTM(input_size=config.filters[config.num_blocks-1], hidden_size=500, batch_first=True, dropout=0.0),
            nn.Linear(in_features=500, out_features=config.num_classes+1)
        ])

    def forward(self, x, edge_data=None):
        output = self.layers[0](x)
        output = output.permute(0,2,1)
        output = self.layers[1](output)[0]
        output = self.layers[2](output)
        output = output.permute(0,2,1)
        return output


_ops = {
    'linear': Linear,
    'conv5': functools.partial(PadConvRelu, kernel_size=5, dilation=1, strides=1, groups=100),
    'conv5d2': functools.partial(PadConvRelu, kernel_size=5, dilation=2, strides=1, groups=100),
    'conv7': functools.partial(PadConvRelu, kernel_size=7, dilation=1, strides=1, groups=100),
    'conv7d2': functools.partial(PadConvRelu, kernel_size=7, dilation=2, strides=1, groups=100),
    'zero': lambda *args, **kwargs: Zero(**kwargs)
}

_branch_ops = {
    0: Zero, # branch not present
    1: Identity # branch present
}

