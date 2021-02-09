# Layers

import torch.nn as nn

def _infer_conv_size(w: int, k: int, s: int, p: int, d: int):
    """Infers the next size after convolution.

    Args:

        w: Input size.

        k: Kernel size.

        s: Stride.

        p: Padding.

        d: Dilation.

    Returns:

        int: Output size.

    """

    x = (w - k - (k - 1) * (d - 1) + 2 * p) // s + 1

    return x


class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def _add_layer(in_channels, out_channels, kernel_size, _input_size, stride=1, padding=0, dilation=1):
        """
        :param out_channels:
        :param kernel_size:
        Filter size
        :param _input_size:
        :param stride:
        :param padding:
        Zero padding.
        :param dilation:
        :return: Convolution, Max Pooling, ReLU, out_channels, out_height, out_width
        """
        in_height, in_width = _input_size[1:]
        conv = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation)
        out_h = _infer_conv_size(in_height, kernel_size, stride, padding, dilation)
        out_w = _infer_conv_size(in_width, kernel_size, stride, padding, dilation)

        relu = nn.ReLU(inplace=True)
        mp = nn.MaxPool2d(2, 2)
        out_h //= 2
        out_w //= 2

        conv_block = nn.Sequential(conv, relu, mp)

        return conv_block, (out_channels, out_h, out_w)

    def _add_layer2(in_channels, out_channels, kernel_size, _input_size, stride=1, padding=0, dilation=1):
        """
        :param out_channels:
        :param kernel_size:
        :param _input_size:
        :param stride:
        :param padding:
        :param dilation:
        :return: Convolution, ReLU, out_channel, out_height, out_width
        """
        in_height, in_width = _input_size[1:]
        conv = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation)
        out_h = _infer_conv_size(in_height, kernel_size, stride, padding, dilation)
        out_w = _infer_conv_size(in_width, kernel_size, stride, padding, dilation)

        relu = nn.ReLU(inplace=True)

        conv_block = nn.Sequential(conv, relu)

        return conv_block, (out_channels, out_h, out_w)

    def _VGG_add_layer(in_channels, out_channels, kernel_size, _input_size, stride=1, padding=1, dilation=1):
        in_height, in_width = _input_size[1:]
        conv1_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation)
        conv1_2 = nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation)

        out_h = _infer_conv_size(in_height, kernel_size, stride, padding, dilation)
        out_w = _infer_conv_size(in_width, kernel_size, stride, padding, dilation)

        relu = nn.ReLU(inplace=True)
        mp = nn.MaxPool2d(2, 2)
        out_h //= 2
        out_w //= 2

        conv_block = nn.Sequential(conv1_1, relu, conv1_2, relu, mp)

        return conv_block, (out_channels, out_h, out_w)

    def _VGG_add_layer2(in_channels, out_channels, kernel_size, _input_size, stride=1, padding=1, dilation=1):
        in_height, in_width = _input_size[1:]
        conv2_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation)
        conv2_2 = nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation)
        conv2_3 = nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation)

        out_h = _infer_conv_size(in_height, kernel_size, stride, padding, dilation)
        out_w = _infer_conv_size(in_width, kernel_size, stride, padding, dilation)

        relu = nn.ReLU(inplace=True)
        mp = nn.MaxPool2d(2, 2)
        out_h //= 2
        out_w //= 2

        conv_block = nn.Sequential(conv2_1, relu, conv2_2, relu, conv2_3, relu, mp)

        return conv_block, (out_channels, out_h, out_w)