import src.utils.interface_tensorboard as tensorboard
import numpy as np
import torch


def pretext_representations(representation, large_count=16, small_count=2):
    large_output = []
    small_output = []
    for i in range(4):
        representation_step = representation[i][0].detach().cpu().numpy()
        temp = []
        for j in range(large_count):
            temp.append(representation_step[0][j])
        large_output.append(np.vstack(temp))
        temp = []
        for j in range(small_count):
            temp.append(representation_step[0][j])
        small_output.append(np.vstack(temp))

    early_output = []
    for i in range(4):
        representation_step = representation[i][1].detach().cpu().numpy()
        early_output.append(representation_step[0])

    return large_output, small_output, early_output


def downstream_representations(representation, large_count=16, small_count=2):
    large_output = []
    small_output = []
    representation_step = representation[0].detach().cpu().numpy()
    for i in range(4):
        temp = []
        for j in range(large_count):
            temp.append(representation_step[i][j])
        large_output.append(np.vstack(temp))
        temp = []
        for j in range(small_count):
            temp.append(representation_step[i][j])
        small_output.append(np.vstack(temp))

    early_output = []
    representation_step = representation[1].detach().cpu().numpy()
    for i in range(4):
        early_output.append(representation_step[i])
    return large_output, small_output, early_output


def make_downstream_target(label, label_dict):
    targets = torch.zeros(len(label)).long()
    for idx in range(len(label)):
        targets[idx] = label_dict[label[idx]]
    return targets


def train_pretext(config, pretext_model, pretext_dataloader, pretext_optimizer, writer, epoch):
    total_loss = 0.0
    pretext_model.train()
    # 1에폭이 지날때마다 moving average로 target network 업데이트
    pretext_model.update_target_weight()

    for batch_idx, (waveform01, waveform02) in enumerate(pretext_dataloader):
        if config['use_cuda']:
            waveform01 = waveform01.cuda()
            waveform02 = waveform02.cuda()

        out_loss, representation = pretext_model(waveform01, waveform02)

        pretext_model.zero_grad()
        out_loss.backward()
        pretext_optimizer.step()

        writer.add_scalar("Pretext_loss/train_step", out_loss, (epoch - 1) * len(pretext_dataloader) + batch_idx)
        total_loss += len(waveform01) * out_loss

        if batch_idx % 20 == 0:
            large_output, small_output, early_output = pretext_representations(representation)
            tensorboard.add_latents_heatmap(
                writer=writer, output=large_output,
                title="pretext-train-large", desc="train-large",
                step=(epoch - 1) * len(pretext_dataloader) + batch_idx)
            tensorboard.add_latents_heatmap(
                writer=writer, output=small_output,
                title="pretext-train-small", desc="train-small",
                step=(epoch - 1) * len(pretext_dataloader) + batch_idx)
            tensorboard.add_latents_heatmap(
                writer=writer, output=early_output,
                title="pretext-train-early", desc="train-early",
                step=(epoch - 1) * len(pretext_dataloader) + batch_idx)

    total_loss /= len(pretext_dataloader.dataset)  # average loss
    writer.add_scalar('Pretext_loss/train_epoch', total_loss, (epoch - 1))
    return total_loss


def train_downstream(config, pretext_model, downstream_model, downstream_dataloader, downstream_criterion,
                     downstream_optimizer, writer, epoch, label_dict):
    total_loss = 0.0
    total_accuracy = 0.0
    pretext_model.eval()
    downstream_model.train()

    for batch_idx, (waveform, label) in enumerate(downstream_dataloader):
        target = make_downstream_target(label, label_dict)
        if config['use_cuda']:
            waveform = waveform.cuda()
            target = target.cuda()

        with torch.no_grad():
            representations = pretext_model.get_representation(waveform)
        if config['downstream_model_name'] == 'DownstreamEarlyClassification':
            representation = representations[1].detach()
        else:
            representation = representations[0].detach()

        prediction = downstream_model(representation)
        out_loss = downstream_criterion(prediction, target)

        downstream_model.zero_grad()
        out_loss.backward()
        downstream_optimizer.step()

        accuracy = torch.zeros(1)
        _, predicted = torch.max(prediction.data, 1)
        total = target.size(0)
        correct = (predicted == target).sum().item()
        accuracy[0] = correct / total

        writer.add_scalar("Downstream_loss/train_step", out_loss,
                          (epoch - 1) * len(downstream_dataloader) + batch_idx)
        total_loss += len(waveform) * out_loss
        writer.add_scalar("Downstream_accuracy/train_step", accuracy * 100,
                          (epoch - 1) * len(downstream_dataloader) + batch_idx)
        total_accuracy += len(waveform) * accuracy

        if batch_idx % 20 == 0:
            large_output, small_output, early_output = downstream_representations(representations)
            tensorboard.add_latents_heatmap(
                writer=writer, output=large_output,
                title="downstream-train-large", desc="train-large",
                step=(epoch - 1) * len(downstream_dataloader) + batch_idx)
            tensorboard.add_latents_heatmap(
                writer=writer, output=small_output,
                title="downstream-train-small", desc="train-small",
                step=(epoch - 1) * len(downstream_dataloader) + batch_idx)
            tensorboard.add_latents_heatmap(
                writer=writer, output=early_output,
                title="downstream-train-early", desc="train-early",
                step=(epoch - 1) * len(downstream_dataloader) + batch_idx)

    total_loss /= len(downstream_dataloader.dataset)  # average loss
    writer.add_scalar('Downstream_loss/train_epoch', total_loss, (epoch - 1))
    total_accuracy /= len(downstream_dataloader.dataset)  # average acc
    writer.add_scalar('Downstream_accuracy/train_epoch', total_accuracy * 100, (epoch - 1))
    return total_loss


def train_downstream_transfer(config, downstream_model, downstream_dataloader, downstream_criterion,
                     downstream_optimizer, writer, epoch, label_dict):
    total_loss = 0.0
    total_accuracy = 0.0
    downstream_model.train()

    for batch_idx, (waveform, label) in enumerate(downstream_dataloader):
        target = make_downstream_target(label, label_dict)
        if config['use_cuda']:
            waveform = waveform.cuda()
            target = target.cuda()

        prediction, representation = downstream_model(waveform)
        out_loss = downstream_criterion(prediction, target)

        downstream_model.zero_grad()
        out_loss.backward()
        downstream_optimizer.step()

        accuracy = torch.zeros(1)
        _, predicted = torch.max(prediction.data, 1)
        total = target.size(0)
        correct = (predicted == target).sum().item()
        accuracy[0] = correct / total

        writer.add_scalar("Downstream_loss/train_step", out_loss,
                          (epoch - 1) * len(downstream_dataloader) + batch_idx)
        total_loss += len(waveform) * out_loss
        writer.add_scalar("Downstream_accuracy/train_step", accuracy * 100,
                          (epoch - 1) * len(downstream_dataloader) + batch_idx)
        total_accuracy += len(waveform) * accuracy

        # representations = representation.detach()
        # if batch_idx % 20 == 0:
        #     large_output, small_output = downstream_representations(representations)
        #     tensorboard.add_latents_heatmap(
        #         writer=writer, output=large_output,
        #         title="Downstream_latent_space", desc="train_large",
        #         step=(epoch - 1) * len(downstream_dataloader) + batch_idx)
        #     tensorboard.add_latents_heatmap(
        #         writer=writer, output=small_output,
        #         title="Downstream_latent_space", desc="train_small",
        #         step=(epoch - 1) * len(downstream_dataloader) + batch_idx)

    total_loss /= len(downstream_dataloader.dataset)  # average loss
    writer.add_scalar('Downstream_loss/train_epoch', total_loss, (epoch - 1))
    total_accuracy /= len(downstream_dataloader.dataset)  # average acc
    writer.add_scalar('Downstream_accuracy/train_epoch', total_accuracy * 100, (epoch - 1))
    return total_loss
