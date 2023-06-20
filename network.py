import torch
import torch.nn as nn
import torch.nn.functional as F


class ResCoreElement(nn.Module):
    """
    Residual Core element used inside the NN. Control the number of filters
    and normalization.
    """

    def __init__(self,
                 input_size,
                 num_filters,
                 ndims=3):
        super(ResCoreElement, self).__init__()
        conv = nn.Conv2d if ndims == 2 else nn.Conv3d
        norm = nn.BatchNorm2d if ndims == 2 else nn.BatchNorm3d
        self.conv1 = conv(input_size,
                          num_filters,
                          kernel_size=3,
                          padding=1)
        self.conv2 = conv(input_size,
                          num_filters,
                          kernel_size=1,
                          padding=0)
        self.bn_add = norm(num_filters) if ndims == 2 else norm(num_filters)

    def forward(self, x):
        """

        """
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_sum = self.bn_add(x_1 + x_2)
        return F.leaky_relu_(x_sum)


class LightClassifier(nn.Module):
    """
    Convolutional layers for classification
    """
    def __init__(self, o_ch, k_filters, activation=False):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=k_filters, out_channels=int(k_filters/4), kernel_size=1,
                                             stride=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=int(k_filters/4), out_channels=int(k_filters/8), kernel_size=1,
                                             stride=1), nn.ReLU())
        self.conv3 = nn.Conv3d(in_channels=int(k_filters/8), out_channels=o_ch, kernel_size=1, stride=1)

        self.activation = activation

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        o = self.conv3(c2)
        if self.activation:
            act = nn.Softmax()
            o = act(o)
        else:
            o = torch.squeeze(o)
        return o


class EncoderForLight(nn.Module):
    def __init__(self, in_ch, base_filters, ndims=3):
        super().__init__()
        self.enc_res1 = ResCoreElement(in_ch, base_filters, ndims=ndims)
        self.enc_res2 = ResCoreElement(1 * base_filters, 2 * base_filters, ndims=ndims)
        self.enc_res3 = ResCoreElement(2 * base_filters, 4 * base_filters, ndims=ndims)
        self.enc_res4 = ResCoreElement(4 * base_filters, 8 * base_filters, ndims=ndims)

        self.pool1 = nn.MaxPool2d(2) if ndims == 2 else nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool2d(2) if ndims == 2 else nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool2d(2) if ndims == 2 else nn.MaxPool3d(2)
        self.gap = nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, x):
        x1 = self.enc_res1(x)
        x = self.pool1(x1)
        x2 = self.enc_res2(x)
        x = self.pool2(x2)
        x3 = self.enc_res3(x)
        x = self.pool3(x3)
        x4 = self.enc_res4(x)
        x = self.gap(x4)

        return x


class LightClassification(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.enc = encoder
        self.clsf = classifier

    def forward(self, x):
        ls_gap = self.enc(x)
        pred = self.clsf(ls_gap)

        return pred
