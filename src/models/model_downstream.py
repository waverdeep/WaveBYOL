import torch.nn as nn
import collections
import torch.nn.functional as F


class DownstreamClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DownstreamClassification, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(input_dim, hidden_dim)),
                    ('bn01', nn.BatchNorm1d(hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, output_dim))

                ]
            )
        )

    def forward(self, x):
        out01 = self.average_pooling(x)
        B, T, D, C = out01.shape
        out01 = out01.reshape((B, T * C * D))

        out02 = self.max_pooling(x)
        B, T, D, C = out02.shape
        out02 = out02.reshape((B, T * C * D))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, p=2)

        output = self.classifier(out_merge)
        return output
