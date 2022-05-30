import torchvision
import torch.nn as nn


def select_feature_encoder_model(model_name, pretrain=True):
    feature_extractor_model = nn.Sequential()

    if model_name == 'h1':
        feature_extractor_model.add_module(
            "feature_extractor_layer",
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        )
    elif model_name == 'h2':
        feature_extractor_model.add_module(
            "feature_extractor_layer",
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),

                nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        )
    elif model_name == 'h3':
        feature_extractor_model.add_module(
            "feature_extractor_layer",
            nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 16, 3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(32, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        )
    elif model_name == 'h4':
        feature_extractor_model.add_module(
            "feature_extractor_layer",
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),

                nn.Conv2d(1024, 2048, 3, stride=1, padding=1),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        )
    elif model_name == 'h5':
        feature_extractor_model.add_module(
            "feature_extractor_layer",
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),

                nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        )
    elif model_name == 'l2':
        feature_extractor_model.add_module(
            "feature_extractor_layer",
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(4, stride=2),

                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(4, stride=2),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(4, stride=2),
            )
        )
    elif model_name == 'l3':
        feature_extractor_model.add_module(
            "feature_extractor_layer",
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),


                nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        )

    return feature_extractor_model