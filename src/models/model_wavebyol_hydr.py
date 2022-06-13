import copy
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import src.models.model_feature_encoder as model_feature_encoder
import torchvision.transforms as transforms

# import torchsummary
# from torchinfo import summary
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def set_requires_grad(model, requires):
    for parameter in model.parameters():
        parameter.requires_grad = requires


def loss_function(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA:
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


def tensor_random_erasing(x, p=0.5):
    length = len(x)
    erasing = transforms.RandomErasing(p=p)
    for i in range(length):
        x[i] = erasing(x[i])
    return x



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
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding, feature_extractor_model, pretrain, config):
        super(Encoder, self).__init__()
        assert (
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.feature_extractor_step = nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(stride, filter_size, padding)):
            self.feature_extractor_step.add_module(
                "encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

        self.feature_encoding_step = model_feature_encoder.select_feature_encoder_model(
            model_name=feature_extractor_model, pretrain=pretrain)

        self.adaptive_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.adaptive_max_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        out = x
        # normalize 먼저 진행 (batch 단위로)
        out = F.normalize(out, dim=-1, p=2)
        out = self.feature_extractor_step(out)
        # chunk and stack layer
        chunks = out.chunk(3, dim=1)
        out_cat = torch.stack(chunks, dim=1)
        # if self.training is True:
        #     out_cat = tensor_random_erasing(out_cat)

        out_cat = F.normalize(out_cat, dim=-1, p=2)
        # feature extract layer
        out_feature = self.feature_encoding_step(out_cat)

        out01 = self.adaptive_max_pooling(out_feature)
        B, T, D, C = out01.shape
        out01 = out01.reshape((B, T * D * C))

        out02 = self.adaptive_average_pooling(out_feature)
        B, T, D, C = out02.shape
        out02 = out02.reshape((B, T * D * C))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=-1, p=2)

        return out_merge, [out_feature, out]


class WaveBYOL(nn.Module):
    def __init__(self, config, encoder_input_dim, encoder_hidden_dim, encoder_filter_size, encoder_stride,
                 encoder_padding, feature_extractor_model, pretrain, mlp_input_dim, mlp_hidden_dim, mlp_output_dim):
        super(WaveBYOL, self).__init__()
        self.config = config

        self.target_ema_updater = EMA(config['ema_decay'])

        self.online_encoder_network = Encoder(
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            filter_size=encoder_filter_size,
            stride=encoder_stride,
            padding=encoder_padding,
            feature_extractor_model=feature_extractor_model,
            pretrain=pretrain,
            config=config,
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

    def get_representation(self, x):
        online, online_rep = self.online_encoder_network(x)
        return online_rep

    def get_early_representation(self, x):
        out = F.normalize(x, dim=-1, p=4)
        online_rep = self.online_encoder_network.encoder(out)
        return online_rep

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
                             target01_representation, target02_representation]


if __name__ == '__main__':
    test_model = WaveBYOL(
        config={"ema_decay": 0.99},
        encoder_input_dim=1,
        encoder_hidden_dim=513,
        encoder_filter_size=[10, 8, 4, 2, 2],
        encoder_stride=[5, 4, 2, 2, 2],
        encoder_padding=[2, 2, 2, 2, 1],
        mlp_input_dim=1024,
        mlp_hidden_dim=4096,
        mlp_output_dim=4096,
        feature_extractor_model="h2",
        pretrain=False
    ).cuda()

    test_data = torch.rand(2, 1, 20480).cuda()
    out_loss, _ = test_model(test_data, test_data)
    pretext_model_params = sum(p.numel() for p in test_model.online_encoder_network.parameters() if p.requires_grad)
    print("encoder parameters: {}".format(pretext_model_params))
    print(out_loss)
    test_model.eval()
    output = test_model.get_representation(test_data)
    print(output[0].size())
    print(output[1].size())
