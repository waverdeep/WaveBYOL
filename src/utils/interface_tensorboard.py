# 텐서보드를 사용해서 Projector를 구현할 때 오류가 있음
# 이 오류를 해결하기 위해서 작성해야 할 것
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import src.utils.interface_file_io as file_io
import numpy as np
import os
import json


# import tensorflow as tf
import librosa
import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# console: tensorboard --logdir=runs --bind_all
# # nohup tensorboard --logdir=runs --bind_all > /dev/null 2>&1
# nohup tensorboard --logdir=runs --host 0.0.0.0 > /dev/null 2>&1 &

def set_tensorboard_writer(name):
    writer = SummaryWriter(name) # 'runs/fashion_mnist_experiment_1'
    return writer


def inspect_model(writer, model, data):
    writer.add_graph(model, data)


def close_tensorboard_writer(writer):
    writer.close()


def add_dataset_figure(writer, dataloader, desc="Train", epoch=0):
    dataiter = iter(dataloader)
    waveform01, waveform02, _, _ = dataiter.next()
    fig = plt.figure()
    plt.plot(waveform01[0].t().numpy(), alpha=0.5)
    plt.plot(waveform02[0].t().numpy(), alpha=0.5)
    plt.title("sample waveform visualization")
    writer.add_figure('Visualize{}'.format(desc), fig, epoch)
    plt.close()


def visualization_dataset_by_byol(writer, dataloader, desc="Train", epoch=0):
    dataiter = iter(dataloader)
    waveform01, waveform02 = dataiter.next()
    fig = plt.figure()
    plt.plot(waveform01[0].t().numpy(), alpha=0.5)
    plt.plot(waveform02[0].t().numpy(), alpha=0.5)
    plt.title("sample waveform visualization")
    writer.add_figure('Visualize{}'.format(desc), fig, epoch)
    plt.close()
    # writer.add_audio('Visualize{}'.format(desc), waveform01[0], epoch, sample_rate=16000)


def add_dataset_figure_by_byol(writer, dataloader, desc="Train", epoch=0):
    dataiter = iter(dataloader)
    waveform01, waveform02 = dataiter.next()
    fig = plt.figure()
    plt.plot(waveform01[0].t().numpy(), alpha=0.5)
    plt.plot(waveform02[0].t().numpy(), alpha=0.5)
    plt.title("sample waveform visualization")
    writer.add_figure('Visualize{}'.format(desc), fig, epoch)
    plt.close()


def add_latent_heatmap(writer, data, title, desc, epoch):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.matshow(data)
    writer.add_figure('{}/{}'.format(title, desc), fig, epoch)
    plt.close()


def add_latents_heatmap(writer, output, title, desc, step):
    fig, axes = plt.subplots(1, 4)
    for i in range(4):
        axes[i].matshow(output[i], aspect='equal')  # , aspect='auto')
    writer.add_figure('{}/{}'.format(title, desc), fig, step)
    plt.close()


def add_confusion_matrix(writer, title, desc, step, label_num, targets, predicts):
    labels = np.arange(label_num)
    output = confusion_matrix(targets, predicts, labels=labels)
    norm_output = output / output.astype(np.float).sum(axis=1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.matshow(norm_output)
    xaxis = np.arange(len(labels))
    ax1.set_xticks(xaxis)
    ax1.set_yticks(xaxis)
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)

    writer.add_figure('{}/{}'.format(title, desc), fig, step)
    plt.close()


def add_classification_matrix(config, epoch, writer, title, desc,step, label_num, targets, predicts):
    labels = list(map(str, list(np.arange(label_num))))
    out = classification_report(targets, predicts, target_names=labels, output_dict=True)
    base_directory = os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name'])
    file_path = os.path.join(base_directory,
                             config['checkpoint_file_name'] + "-model-best-epoch-{}-clf.json".format(epoch))
    if not os.path.exists(os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name'])):
        file_io.make_directory(os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name']))
    with open(file_path, 'w') as f:
        json.dump(out, f)

    x1 = np.arange(1, label_num * 2, 2)
    x2 = x1 + 0.5
    x3 = x1 + 1

    precision_value = []
    recall_value = []
    f1_value = []

    for data in labels:
        precision_value.append(out[data]['precision'])
        recall_value.append(out[data]['recall'])
        f1_value.append(out[data]['f1-score'])

    fig = plt.figure()
    plt.bar(x1, precision_value, color='r', width=0.5, label='precision')
    plt.bar(x2, recall_value, color='g', width=0.5, label='recall')
    plt.bar(x3, f1_value, color='b', width=0.5, label='fl-score')
    plt.xticks(x2, labels)
    plt.legend()

    writer.add_figure('{}/{}'.format(title, desc), fig, step)
    plt.close()

def add_classification_avg_matrix(writer, title, desc, step, label_num, targets, predicts):
    labels = np.arange(label_num)
    term_labels = ['macro avg', 'weighted avg']
    out = classification_report(targets, predicts, target_names=labels, output_dict=True)

    x1 = [1, 3]
    x2 = [1.5, 3.5]
    x3 = [2, 4]

    precision_value = []
    recall_value = []
    f1_value = []
    for data in term_labels:
        precision_value.append(out[data]['precision'])
        recall_value.append(out[data]['recall'])
        f1_value.append(out[data]['f1-score'])

    fig = plt.figure()
    plt.bar(x1, precision_value, color='r', width=0.5, label='precision')
    plt.bar(x2, recall_value, color='g', width=0.5, label='recall')
    plt.bar(x3, f1_value, color='b', width=0.5, label='fl-score')
    plt.xticks(x2, term_labels)
    plt.legend()

    writer.add_figure('{}/{}'.format(title, desc), fig, step)
    plt.close()
