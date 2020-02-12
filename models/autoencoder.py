import torch.nn.functional as F
import torch.nn as nn
import models
import torch

class DecodeBlock(nn.Module):
    expansion = 1
    activation_fns = {'relu': nn.ReLU(inplace=True), 'sig': nn.Sigmoid()}

    def __init__(self, inplanes, planes, scale=2, out_shape=None, act='relu', mode='bilinear', inlayer=False, outlayer=False, skip_conn=False):
        super(DecodeBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=1, padding=1)
        self.out_shape, self.scale, self.mode = out_shape, scale, mode
        self.act = self.activation_fns[act]
        self.BN = None
        self.outlayer = outlayer
        self.inlayer = inlayer
        self.skip_conn = skip_conn
        self.skipx = None
        if act == 'relu':
            self.BN = nn.BatchNorm2d(planes)

    def forward(self, input):
        x = input
        skip_conn = None

        # On the input block and intermediate blocks, store the skip connection
        if self.skip_conn:
            x, skip_conn = input[0], input[1]

        # Scaling is only performed on the input layer
        if self.inlayer:
            x = nn.functional.interpolate(x, size=self.out_shape, scale_factor=self.scale,
                                          mode=self.mode, align_corners=True)

        x = self.conv(x)

        # Batch Norm only on relu (otherwise, is None)
        if self.BN is not None:
            x = self.BN(x)

        # If this is an output layer and we have a skip connection, add it
        if self.outlayer and self.skip_conn:
            x += skip_conn

        # Activate
        if not self.outlayer and self.skip_conn:
            out = self.act(x), skip_conn
        else:
            out = self.act(x)

        return out


class Autoencoder(nn.Module):
    # Store the model layer count for each depth of resnet (so we can construct the inverse architecture)
    model_layers = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3],
                    'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3],
                    'resnet152': [3, 8, 36, 3]}

    def __init__(self, arch='resnet18', low_dim=128, skip_conn=False):
        super(Autoencoder, self).__init__()
        self.inplanes = low_dim
        self.skip_conn = skip_conn
        # Create the specified resnet as the encoder
        self.encoder = create_model(arch, low_dim)

        # Get the number of blocks per layer (inverted)
        decode_layers = self.model_layers[arch][::-1]
        # Scale tensor to direct multiple of original (224) image
        self.decode5 = self._make_decode_layer(DecodeBlock, planes=low_dim, blocks=1, scale=None,
                                               out_shape=(7, 7),
                                               act='relu')
        self.decode4 = self._make_decode_layer(DecodeBlock, planes=256, blocks=decode_layers[3], scale=2, act='relu', skip_conn=skip_conn)
        nxt_output = self.inplanes // 2
        self.decode3 = self._make_decode_layer(DecodeBlock, planes=nxt_output, blocks=decode_layers[2], scale=2, act='relu', skip_conn=skip_conn)
        nxt_output = nxt_output // 2

        # Scale tensor to direct multiple of original (224) image
        self.decode2 = self._make_decode_layer(DecodeBlock, planes=nxt_output, blocks=decode_layers[1], scale=None, out_shape=(56, 56),
                                       act='relu', skip_conn=skip_conn)

        # Don't scale number of channels down (symmetric to resnet, which goes from 3 => 64 channels)
        self.decode1 = self._make_decode_layer(DecodeBlock, planes=nxt_output, blocks=decode_layers[0], scale=2, act='relu', skip_conn=skip_conn)

        # Perform final reduction to 3 channels
        self.decode0 = self._make_decode_layer(DecodeBlock, planes=3, blocks=1, scale=2, act='sig', skip_conn=False)

    def forward(self, x):
        # This was implemented using based on the description here: https://arxiv.org/pdf/1606.08921.pdf
        # Accumulate the outputs from each layer (this includes initial conv filters)
        skipx = []
        skipx_targets = ['relu', 'layer2', 'layer3', 'layer4']
        # Feed the input through the encoder, collecting skip connections
        curr_pos = None

        for module_pos, module in self.encoder._modules.items():
            if curr_pos is None:
                curr_pos = module_pos
            # Don't add the output of avg pool / FC to skip connection list
            if curr_pos != module_pos and module_pos in skipx_targets:
                skipx.append(x)
                curr_pos = module_pos
            # Reshape the output of the avg pooling layer
            if module_pos == 'fc':
                x = x.view(x.size(0), -1)
            x = module(x)

        # Reshape the output so that it matches
        x = torch.reshape(x, (x.size(0), x.size(1), 1, 1))
        # Following the example in the the paper, a skip connection occurs ever 2 conv layers
        # The first and last convolutional layers are the 'odd man out' so to speak, since they only contain 1 conv

        # Scale the output dimensions from (_, 128, 1, 1) => (_ , 128, 7, 7) so we can add the skip_connection in layer4
        x = self.decode5(x)
        # Scale the output channels/dimensions from (_, 128, 1, 1) => (_ , 256, 14, 14)
        x = self.decode4({0: x, 1: skipx[-1]}) if self.skip_conn else self.decode4(x)
        # Scale the output channels/dimensions from (_, 256, 1, 1) => (_ , 128, 28, 28)
        x = self.decode3({0: x, 1: skipx[-2]}) if self.skip_conn else self.decode3(x)
        # Scale the output channels/dimensions from (_, 128, 1, 1) => (_ , 64, 56, 56)
        x = self.decode2({0: x, 1: skipx[-3]}) if self.skip_conn else self.decode2(x)
        # Scale the output channels/dimensions from (_ , 64, 56, 56) => (_ , 64, 112, 112)
        x = self.decode1({0: x, 1: skipx[-4]}) if self.skip_conn else self.decode1(x)
        # Scale the output channels/dimensions from (_ , 64, 112, 112) => (_ , 3, 224, 224)
        x = self.decode0(x)

        return x

    def _make_decode_layer(self, block, planes, blocks, scale=2, out_shape=None, act='relu', mode='bilinear', skip_conn=False):
        layers = []
        layers.append(block(self.inplanes, planes, scale=scale, out_shape=out_shape, act=act, mode=mode, inlayer=True, skip_conn=skip_conn))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, scale=1, outlayer=(i == blocks-1), skip_conn=skip_conn))

        return nn.Sequential(*layers)

def create_model(arch='resnet18', low_dim=128):
    return models.__dict__[arch](low_dim=low_dim)

