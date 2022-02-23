import copy
import os
import collections
import torch
import torchvision
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding):
        super(Encoder, self).__init__()
        assert(
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.encoder = nn.Sequential()
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))

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

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 2, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.unsqueeze(1)
        out = self.features(out)

        out01 = self.average_pooling(out)
        B, T, D, C = out01.shape
        out01 = out01.reshape((B, T * D * C))

        out02 = self.max_pooling(out)
        B, T, D, C = out02.shape
        out02 = out02.reshape((B, T * D * C))

        out_merge = out01 + out02

        return out_merge, out


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding, output_padding):
        super(Decoder, self).__init__()
        assert(
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.decoder = nn.Sequential()

        for index, (stride, filter_size, padding, output_padding) in enumerate(zip(stride, filter_size, padding, output_padding)):
            self.decoder.add_module(
                "encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding, output_padding=output_padding),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

        self.features = nn.Sequential(

            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, 5, stride=3, padding=2, output_padding=(0, 2)),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.squeeze(1)
        out = self.decoder(out)

        return out


class WaveBYOLCombine(nn.Module):
    def __init__(self, config, encoder_input_dim, encoder_hidden_dim, encoder_filter_size,
                 encoder_stride, encoder_padding, decoder_output_padding, mlp_input_dim, mlp_hidden_dim,
                 mlp_output_dim):
        super(WaveBYOLCombine, self).__init__()
        self.config = config

        self.target_ema_updater = EMA(config['ema_decay'])

        self.online_encoder_network = Encoder(
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            filter_size=encoder_filter_size,
            stride=encoder_stride,
            padding=encoder_padding,
        )

        reversed_filter_size = list(reversed(encoder_filter_size))
        reversed_stride = list(reversed(encoder_stride))
        reversed_padding = list(reversed(encoder_padding))

        self.online_decoder_network = Decoder(
            input_dim=encoder_hidden_dim,
            hidden_dim=encoder_input_dim,
            filter_size=reversed_filter_size,
            stride=reversed_stride,
            padding=reversed_padding,
            output_padding=decoder_output_padding
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

        self.decoded_loss = nn.L1Loss()

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

        online01_decoded = self.online_decoder_network(online01_representation)
        online02_decoded = self.online_decoder_network(online02_representation)

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
        loss = loss.mean()

        loss01_decoded = self.decoded_loss(online01_decoded, waveform01)
        loss02_decoded = self.decoded_loss(online02_decoded, waveform02)
        loss_decoded = loss01_decoded + loss02_decoded
        loss_decoded = loss_decoded.mean()

        return loss + loss_decoded, [online01_representation, online02_representation,
                             target01_representation.detach(), target02_representation.detach()]


if __name__ == '__main__':
    test_model = WaveBYOLCombine(
        config={"ema_decay": 0.99},
        encoder_input_dim=1,
        encoder_hidden_dim=256,
        encoder_filter_size=[10, 3, 3, 3, 3, 2, 2],
        encoder_stride=[5, 2, 2, 2, 2, 2, 2],
        encoder_padding=[2, 2, 2, 2, 2, 2, 1],
        decoder_output_padding=[1, 0, 1, 1, 0, 0, 4],
        mlp_input_dim=128,
        mlp_hidden_dim=4096,
        mlp_output_dim=4096,
    ).cuda()

    test_data01 = torch.rand(2, 1, 20480).cuda()
    test_data02 = torch.rand(2, 1, 20480).cuda()

    out_loss, rep = test_model(test_data01, test_data02)
    print(out_loss)
    print(rep[0].size())
    rep = rep[0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4)
    axes[0].matshow(rep[0][0], aspect='equal')  # , aspect='auto')
    axes[1].matshow(rep[0][0], aspect='equal')  # , aspect='auto')
    axes[2].matshow(rep[0][0], aspect='equal')  # , aspect='auto')
    axes[3].matshow(rep[0][0], aspect='equal')
    plt.show()





    # encoder_model = Encoder(
    #     input_dim=1,
    #     hidden_dim=512,
    #     stride=[5, 2, 2, 2, 2, 2, 2],
    #     filter_size=[10, 3, 3, 3, 3, 2, 2],
    #     padding=[2, 2, 2, 2, 2, 2, 1],
    #     # stride=[5, 4, 2, 2, 2],
    #     # filter_size=[10, 8, 4, 4, 4],
    #     # padding=[2, 2, 2, 2, 1],
    # ).cuda()
    #
    # decoder_model = Decoder(
    #     input_dim=512,
    #     hidden_dim=1,
    #     stride=[2, 2, 2, 2, 2, 2, 5],
    #     filter_size=[2, 2, 3, 3, 3, 3, 10],
    #     padding=[1, 2, 2, 2, 2, 2, 2],
    #     output_padding=[1, 0, 1, 1, 0, 0, 4],
    # ).cuda()
    #
    # test_data = torch.rand(2, 1, 20480).cuda()
    #
    # output = encoder_model(test_data)
    # output = decoder_model(output)