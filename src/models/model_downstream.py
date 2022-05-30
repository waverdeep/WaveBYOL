import torch.nn as nn
import collections
import torch.nn.functional as F
import torch


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
                    ('linear02', nn.Linear(hidden_dim, output_dim)),
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
        out_merge = F.normalize(out_merge, dim=-1, p=2)
        # print(out_merge.size())

        output = self.classifier(out_merge)
        return output


class DownstreamEarlyClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DownstreamEarlyClassification, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.average_pooling = nn.AdaptiveAvgPool1d(1)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(input_dim, hidden_dim)),
                    ('bn01', nn.BatchNorm1d(hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, output_dim)),
                ]
            )
        )

    def forward(self, x):
        out01 = self.average_pooling(x)
        B, T, D = out01.shape
        out01 = out01.reshape((B, T * D))

        out02 = self.max_pooling(x)
        B, T, D = out02.shape
        out02 = out02.reshape((B, T * D))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=-1, p=2)

        output = self.classifier(out_merge)
        return output


class DownstreamFlatClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DownstreamFlatClassification, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.average_pooling = nn.AdaptiveAvgPool2d((10, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((10, 1))

        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(input_dim, hidden_dim)),
                    ('bn01', nn.BatchNorm1d(hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, output_dim)),
                ]
            )
        )
    def get_embedding(self, x):
        out01 = self.average_pooling(x)
        B, T, C, D = out01.shape
        out01 = out01.reshape((B, T, C * D))

        out02 = self.max_pooling(x)
        B, T, C, D = out02.shape
        out02 = out02.reshape((B, T, C * D))

        out_merge = out01 + out02

        return out_merge

    def forward(self, x):
        out01 = self.average_pooling(x)
        B, T, C, D = out01.shape
        out01 = out01.reshape((B, T * C * D))

        out02 = self.max_pooling(x)
        B, T, C, D = out02.shape
        out02 = out02.reshape((B, T * C * D))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=-1, p=2)

        output = self.classifier(out_merge)
        return output


class DownstreamFlatEmbeddingClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DownstreamFlatEmbeddingClassification, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.average_pooling = nn.AdaptiveAvgPool2d((10, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((10, 1))

        self.embedding = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(input_dim, hidden_dim)),
                    ('bn01', nn.BatchNorm1d(hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, hidden_dim)),
                ]
            )
        )

        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(hidden_dim, hidden_dim)),
                    ('bn01', nn.BatchNorm1d(hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, output_dim)),
                ]
            )
        )

    def get_embedding(self, x):
        out_x = self.embedding(x)
        print(out_x.size())
        out01 = self.average_pooling(out_x)
        B, T, C, D = out01.shape
        out01 = out01.reshape((B, T * C * D))

        out02 = self.max_pooling(out_x)
        B, T, C, D = out02.shape
        out02 = out02.reshape((B, T * C * D))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=-1, p=2)
        embedding = self.embedding(out_merge)
        return embedding

    def forward(self, x):
        out_x = self.embedding(x)
        out01 = self.average_pooling(out_x)
        B, T, C, D = out01.shape
        out01 = out01.reshape((B, T * C * D))

        out02 = self.max_pooling(out_x)
        B, T, C, D = out02.shape
        out02 = out02.reshape((B, T * C * D))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=-1, p=2)

        embedding = self.embedding(out_merge)
        output = self.classifier(embedding)
        return output


class DownstreamFlatTransferClassification(nn.Module):
    def __init__(self, pretext_model, input_dim, hidden_dim, output_dim):
        super(DownstreamFlatTransferClassification, self).__init__()

        self.pretext_model = pretext_model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.average_pooling = nn.AdaptiveAvgPool2d((10, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((10, 1))

        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(input_dim, hidden_dim)),
                    ('bn01', nn.BatchNorm1d(hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, output_dim)),
                ]
            )
        )

    def forward(self, x):
        out, _ = self.pretext_model.get_representation(x)
        out01 = self.average_pooling(out)
        B, T, C, D = out01.shape
        out01 = out01.reshape((B, T * C * D))
        out02 = self.max_pooling(out)
        B, T, C, D = out02.shape
        out02 = out02.reshape((B, T * C * D))
        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=-1, p=2)
        output = self.classifier(out_merge)
        return output, out


class DownstreamFlatEmbeddingTransferClassification(nn.Module):
    def __init__(self, pretext_model, input_dim, hidden_dim, output_dim):
        super(DownstreamFlatEmbeddingTransferClassification, self).__init__()
        self.pretext_model = pretext_model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.average_pooling = nn.AdaptiveAvgPool2d((10, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((10, 1))

        self.embedding = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(input_dim, hidden_dim)),
                    ('bn01', nn.BatchNorm1d(hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, hidden_dim)),
                ]
            )
        )

        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(hidden_dim, hidden_dim)),
                    ('bn01', nn.BatchNorm1d(hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, output_dim)),
                ]
            )
        )

    def get_embedding(self, x):
        out_x, _ = self.pretext_model.get_representation(x)
        out01 = self.average_pooling(out_x)
        B, T, C, D = out01.shape
        out01 = out01.reshape((B, T * C * D))

        out02 = self.max_pooling(out_x)
        B, T, C, D = out02.shape
        out02 = out02.reshape((B, T * C * D))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=-1, p=2)
        # print(out_merge.size())

        embedding = self.embedding(out_merge)
        return embedding

    def forward(self, x):
        out_x, _ = self.pretext_model.get_representation(x)
        # print(out_x.size())
        out01 = self.average_pooling(out_x)
        B, T, C, D = out01.shape
        out01 = out01.reshape((B, T * C * D))

        out02 = self.max_pooling(out_x)
        B, T, C, D = out02.shape
        out02 = out02.reshape((B, T * C * D))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=-1, p=2)
        # print(out_merge.size())

        embedding = self.embedding(out_merge)
        output = self.classifier(embedding)
        return output, [out_x, _]


if __name__ == '__main__':
    test_model = DownstreamFlatEmbeddingClassification(
        input_dim=10240,
        hidden_dim=4096,
        output_dim=1251
    )

    input_data = torch.rand(2, 1024, 10 ,8)
    output_data = test_model(input_data)
    print(output_data.size())

