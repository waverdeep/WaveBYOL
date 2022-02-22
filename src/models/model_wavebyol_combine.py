import copy
import os
import collections
import torch
import torchvision
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def set_requires_grad(model, requires):
    for parameter in model.parameters():
        parameter.requires_grad = requires


def loss_function(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding, combine_model_name,
                 linear_input_dim, linear_hidden_dim):
        super(Encoder, self).__init__()
        self.combine_model_name = combine_model_name
        self.linear_input_dim = linear_input_dim
        self.linear_hidden_dim = linear_hidden_dim
        assert(
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.encoder = nn.Sequential()
        self.encoder_combine = nn.Sequential()
        self.encoder_efficient = None
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.full_connected_network = nn.Sequential()

        for index, (stride, filter_size, padding) in enumerate(zip(stride, filter_size, padding)):
            self.encoder.add_module(
                "encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

        if self.combine_model_name == "resnet50":
            self.encoder_combine.add_module(
                "combine_network",
                nn.Sequential(
                    nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                    torchvision.models.resnet50(pretrained=True).conv1,
                    torchvision.models.resnet50(pretrained=True).bn1,
                    torchvision.models.resnet50(pretrained=True).relu,
                    # maxpool layer는 사용하지 않고 작업을 해보려고 함.
                    torchvision.models.resnet50(pretrained=True).layer1,
                    torchvision.models.resnet50(pretrained=True).layer2,
                    torchvision.models.resnet50(pretrained=True).layer3,
                    torchvision.models.resnet50(pretrained=True).layer4,
                )
            )
        elif self.combine_model_name == "resnet152":
            self.encoder_combine.add_module(
                "combine_network",
                nn.Sequential(
                    nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                    torchvision.models.resnet152(pretrained=True).conv1,
                    torchvision.models.resnet152(pretrained=True).bn1,
                    torchvision.models.resnet152(pretrained=True).relu,
                    # maxpool layer는 사용하지 않고 작업을 해보려고 함.
                    torchvision.models.resnet152(pretrained=True).layer1,
                    torchvision.models.resnet152(pretrained=True).layer2,
                    torchvision.models.resnet152(pretrained=True).layer3,
                    torchvision.models.resnet152(pretrained=True).layer4,
                )
            )
        elif self.combine_model_name == "mobilenetv2":
            self.encoder_combine.add_module(
                "combine_network",
                nn.Sequential(
                    nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                    torchvision.models.mobilenet_v2(pretrained=True).features
                )
            )
        elif self.combine_model_name == "mobilenetv3_large":
            self.encoder_combine.add_module(
                "combine_network",
                nn.Sequential(
                    nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                    torchvision.models.mobilenet_v3_large(pretrained=True).features
                )
            )
        elif "efficientnet" in self.combine_model_name:
            self.encoder_combine.add_module(
                "combine_network",
                nn.Sequential(
                    nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                )
            )
            if "no" in combine_model_name:
                self.model_name = self.model_name.replace('no', "")
                self.encoder_efficient = EfficientNet.from_name("{}".format(self.combine_model_name))
            else:
                self.encoder_efficient = EfficientNet.from_pretrained("{}".format(self.combine_model_name))

        self.full_connected_network = nn.Sequential(
            nn.Linear(self.linear_input_dim, self.linear_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.linear_hidden_dim, self.linear_hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.unsqueeze(1)
        out = self.encoder_combine(out)
        if "efficientnet" in self.combine_model_name:
            out = self.encoder_efficient(out)

        out01 = self.average_pooling(out)
        B, T, D, C =  out01.shape
        out01 = out01.reshape((B, T*D*C))

        out02 = self.max_pooling(out)
        B, T, D, C = out02.shape
        out02 = out02.reshape((B, T*D*C))

        out_merge = out01 + out02
        out_merge = self.full_connected_network(out_merge)
        return out_merge, out


class WaveBYOLCombine(nn.Module):
    def __init__(self, config, encoder_input_dim, encoder_hidden_dim, encoder_filter_size,
                 encoder_stride, encoder_padding, linear_input_dim, linear_hidden_dim,
                 mlp_input_dim, mlp_hidden_dim, mlp_output_dim, combine_model_name):
        super(WaveBYOLCombine, self).__init__()
        self.config = config

        self.target_ema_updater = EMA(config['ema_decay'])

        self.online_encoder_network = Encoder(
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            filter_size=encoder_filter_size,
            stride=encoder_stride,
            padding=encoder_padding,
            combine_model_name=combine_model_name,
            linear_input_dim=linear_input_dim,
            linear_hidden_dim=linear_hidden_dim
        )

        self.online_projector_network = MLPNetwork(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_output_dim
        )

        self.online_predictor_network = MLPNetwork(
            input_dim=mlp_hidden_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_output_dim
        )

        self.target_encoder_network = None
        self.target_projector_network = None

    def get_target_network(self):
        self.target_encoder_network = copy.deepcopy(self.online_encoder_network)
        set_requires_grad(self.target_encoder_network, False)
        self.target_projector_network = copy.deepcopy(self.online_projector_network)
        set_requires_grad(self.target_projector_network, False)

    def update_target_weight(self):
        if self.target_encoder_network is None:
            self.get_target_network()
        update_moving_average(self.target_ema_updater, self.target_encoder_network, self.online_encoder_network)
        update_moving_average(self.target_ema_updater, self.target_projector_network, self.online_projector_network)

    def forward(self, waveform01, waveform02):
        online01, online01_representation = self.online_encoder_network(waveform01)
        online02, online02_representation = self.online_encoder_network(waveform02)

        online01_project = self.online_projector_network(online01)
        online02_project = self.online_projector_network(online02)

        online01_predict = self.online_predictor_network(online01_project)
        online02_predict = self.online_predictor_network(online02_project)

        with torch.no_grad():
            if self.target_encoder_network is None:
                self.get_target_network()
            target01, target01_representation = self.target_encoder_network(waveform01)
            target02, target02_representation = self.target_encoder_network(waveform02)

            target01_project = self.target_projector_network(target01)
            target02_project = self.target_projector_network(target02)

        loss01 = loss_function(online01_predict, target02_project.detach())
        loss02 = loss_function(online02_predict, target01_project.detach())
        loss = loss01 + loss02

        return loss.mean(), [online01_representation, online02_representation,
                             target01_representation.detach(),target02_representation.detach()]


if __name__ == '__main__':
    test_model = WaveBYOLCombine(
        config={"ema_decay":0.99},
        encoder_input_dim=1,
        encoder_hidden_dim=512,
        encoder_filter_size=[10, 8, 4, 4],
        encoder_stride=[5, 4, 2, 2],
        encoder_padding=[2, 2, 2, 1],
        linear_input_dim=2048,
        linear_hidden_dim=2048,
        mlp_input_dim=2048,
        mlp_hidden_dim=4096,
        mlp_output_dim=4096,
        combine_model_name="resnet50"
    ).cuda()

    test_data = torch.rand(2, 1, 20480).cuda()

    out_loss, _ = test_model(test_data, test_data)
    print(out_loss)



