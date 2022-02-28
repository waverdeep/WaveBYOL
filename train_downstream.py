import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import src.data.dataset as dataset
import src.models.model as model_pack
import src.optimizers.optimizer as optimizers
import src.utils.interface_tensorboard as tensorboard
import src.utils.interface_train_tool as train_tool
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # Configuration 불러오기
    parser = argparse.ArgumentParser(description='waverdeep - downstream task <downstream-classification>')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument('--configuration', required=False,
                        default='./config/T03-ravdess-WaveBYOL-Original-20480.json')
    args = parser.parse_args()
    now = train_tool.setup_timestamp()

    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    print(">>> Train Downstream - SpecBYOL <<<")
    print("Use GPU: ", torch.cuda.is_available())
    print(config)

    print("load dataset...")
    # setup train/test dataloader
    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, test_dataset = dataset.get_dataloader(config=config, mode='test')

    print("load model...")
    pretext_model = model_pack.load_model(config, config['pretext_model_name'], config['pretext_checkpoint'])
    downstream_model = model_pack.load_model(config, config['downstream_model_name'], config['downstream_checkpoint'])

    # setup speaker classfication label
    acoustic_dict = train_dataset.acoustic_dict
    print("speaker_num: {}".format(len(acoustic_dict.keys())))

    optimizer = optimizers.get_optimizer(downstream_model.parameters(), config)

    # if gpu available: load gpu
    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        downstream_model = downstream_model.cuda()

    writer = tensorboard.set_tensorboard_writer(
        "{}-{}".format(config['tensorboard_writer_name'], now)
    )

    # print model information
    print("model structure...")
    model_params = sum(p.numel() for p in pretext_model.parameters() if p.requires_grad)
    print("model parameters: {}".format(model_params))
    print("{}".format(pretext_model))

    print(">>> downstream_model_structure <<<")
    model_params = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
    print("downstream model parameters: {}".format(model_params))
    print("{}".format(downstream_model))

    # start training
    best_accuracy = 0.0
    num_of_epoch = config['epoch']
    for epoch in range(num_of_epoch):
        epoch = epoch + 1
        print("start train ... [ {}/{} epoch - {} iter  ]".format(epoch, num_of_epoch, len(train_loader)))
        train_accuracy, train_loss = train(config, writer, epoch, pretext_model, downstream_model, train_loader,
                                           optimizer, acoustic_dict)
        print("start test ... [ {}/{} epoch - {} iter  ]".format(epoch, num_of_epoch, len(train_loader)))
        test_accuracy, test_loss = test(config, writer, epoch, pretext_model, downstream_model,
                                        test_loader, acoustic_dict)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            train_tool.save_checkpoint(config=config, model=downstream_model, optimizer=optimizer,
                                       loss=test_loss, epoch=best_epoch, mode="best",
                                       date='{}'.format(now))


def train(config, writer, epoch, pretext_model, downstream_model, data_loader, optimizer, acoustic_dict):
    total_loss = 0.0
    total_accuracy = 0.0
    pretext_model.eval()
    downstream_model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (waveform, acoustic_id) in enumerate(data_loader):
        # 데이터로더의 변경이 필요한가?
        targets = make_target(acoustic_id, acoustic_dict)
        if config['use_cuda']:
            data = waveform.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            representation = pretext_model.get_representation(data)
        representation = representation.detach()
        # print(representation.size())
        predictions = downstream_model(representation)
        loss = criterion(predictions, targets)

        downstream_model.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.zeros(1)

        _, predicted = torch.max(predictions.data, 1)

        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy[0] = correct / total

        writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(data_loader) + batch_idx)
        writer.add_scalar('Accuracy/train_step', accuracy * 100, (epoch - 1) * len(data_loader) + batch_idx)
        total_loss += len(data) * loss
        total_accuracy += len(data) * accuracy

        if batch_idx % 20 == 0:
            output = representation.detach()
            output = output.cpu().numpy()
            outputs = []
            for j in range(4):
                temp = []  # np.vstack()
                for i in range(16):
                    temp.append(output[j][i])
                outputs.append(np.vstack(temp))

            tensorboard.add_train_latent_heatmap(writer, outputs[0], outputs[1], outputs[2], outputs[3],
                                                 "TrainLatentSpace",
                                                 "Large",
                                                 (epoch - 1) * len(data_loader) + batch_idx
                                                 )
            outputs = []
            for j in range(4):
                temp = []  # np.vstack()
                for i in range(2):
                    temp.append(output[j][i])
                outputs.append(np.vstack(temp))

            tensorboard.add_train_latent_heatmap(writer, outputs[0], outputs[1], outputs[2], outputs[3],
                                                 "TrainLatentSpace",
                                                 "Small",
                                                 (epoch - 1) * len(data_loader) + batch_idx
                                                 )

    total_loss /= len(data_loader.dataset)  # average loss
    total_accuracy /= len(data_loader.dataset)  # average acc

    writer.add_scalar('Loss/train', total_loss, (epoch - 1))
    writer.add_scalar('Accuracy/train', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss


def test(config, writer, epoch, pretext_model, downstream_model, data_loader, acoustic_dict):
    total_loss = 0.0
    total_accuracy = 0.0
    pretext_model.eval()
    downstream_model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (waveform, acoustic_id) in enumerate(data_loader):
            # 데이터로더의 변경이 필요한가?
            targets = make_target(acoustic_id, acoustic_dict)
            if config['use_cuda']:
                data = waveform.cuda()
                targets = targets.cuda()

            representation = pretext_model.get_representation(data)
            predictions = downstream_model(representation)
            loss = criterion(predictions, targets)

            accuracy = torch.zeros(1)
            _, predicted = torch.max(predictions.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy[0] = correct / total
            print("targets: ", targets)
            print("predcts: ", predicted)
            if batch_idx % 20 == 0:
                output = representation.detach()
                output = output.cpu().numpy()
                outputs = []
                for j in range(4):
                    temp = []  # np.vstack()
                    for i in range(16):
                        temp.append(output[j][i])
                    outputs.append(np.vstack(temp))

                tensorboard.add_train_latent_heatmap(writer, outputs[0], outputs[1], outputs[2], outputs[3],
                                                     "TestLatentSpace",
                                                     "Large",
                                                     (epoch - 1) * len(data_loader) + batch_idx
                                                     )
                outputs = []
                for j in range(4):
                    temp = []  # np.vstack()
                    for i in range(2):
                        temp.append(output[j][i])
                    outputs.append(np.vstack(temp))

                tensorboard.add_train_latent_heatmap(writer, outputs[0], outputs[1], outputs[2], outputs[3],
                                                     "TestLatentSpace",
                                                     "Small",
                                                     (epoch - 1) * len(data_loader) + batch_idx
                                                     )

            writer.add_scalar('Loss/test_step', loss, (epoch - 1) * len(data_loader) + batch_idx)
            writer.add_scalar('Accuracy/test_step', accuracy * 100, (epoch - 1) * len(data_loader) + batch_idx)
            total_loss += len(data) * loss
            total_accuracy += len(data) * accuracy


        total_loss /= len(data_loader.dataset)  # average loss
        total_accuracy /= len(data_loader.dataset)  # average acc

        writer.add_scalar('Loss/test', total_loss, (epoch - 1))
        writer.add_scalar('Accuracy/test', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss


def make_target(speaker_id, speaker_dict):
    targets = torch.zeros(len(speaker_id)).long()
    for idx in range(len(speaker_id)):
        targets[idx] = speaker_dict[speaker_id[idx]]
    return targets


if __name__ == '__main__':
    main()
