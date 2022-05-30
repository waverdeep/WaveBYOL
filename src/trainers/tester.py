import src.utils.interface_tensorboard as tensorboard
import src.trainers.trainer as trainer
import torch


def test_pretext(config, pretext_model, pretext_dataloader, writer, epoch):
    total_loss = 0.0
    pretext_model.eval()
    with torch.no_grad():
        for batch_idx, (waveform01, waveform02) in enumerate(pretext_dataloader):
            if config['use_cuda']:
                waveform01 = waveform01.cuda()
                waveform02 = waveform02.cuda()

            out_loss, representation = pretext_model(waveform01, waveform02)

            writer.add_scalar("Pretext_loss/test_step", out_loss, (epoch - 1) * len(pretext_dataloader) + batch_idx)
            total_loss += len(waveform01) * out_loss

            if batch_idx % 20 == 0:
                large_output, small_output, early_output = trainer.pretext_representations(representation)
                tensorboard.add_latents_heatmap(
                    writer=writer, output=large_output,
                    title="pretext-test-large", desc="test-large",
                    step=(epoch - 1) * len(pretext_dataloader) + batch_idx)
                tensorboard.add_latents_heatmap(
                    writer=writer, output=small_output,
                    title="pretext-test-small", desc="test-small",
                    step=(epoch - 1) * len(pretext_dataloader) + batch_idx)
                tensorboard.add_latents_heatmap(
                    writer=writer, output=early_output,
                    title="pretext-test-early", desc="test-early",
                    step=(epoch - 1) * len(pretext_dataloader) + batch_idx)

    total_loss /= len(pretext_dataloader.dataset)  # average loss
    writer.add_scalar('Pretext_loss/test_epoch', total_loss, (epoch - 1))
    return total_loss


def test_downstream(config, pretext_model, downstream_model, downstream_dataloader, downstream_criterion,
                    writer, epoch, label_dict):
    total_loss = 0.0
    total_accuracy = 0.0
    total_target = []
    total_predict = []
    pretext_model.eval()
    downstream_model.eval()
    with torch.no_grad():
        for batch_idx, (waveform, label) in enumerate(downstream_dataloader):
            target = trainer.make_downstream_target(label, label_dict)

            if config['use_cuda']:
                waveform = waveform.cuda()
                target = target.cuda()

            representations = pretext_model.get_representation(waveform)
            if config['downstream_model_name'] == 'DownstreamEarlyClassification':
                representation = representations[1].detach()
            else:
                representation = representations[0].detach()
            prediction = downstream_model(representation)
            out_loss = downstream_criterion(prediction, target)

            accuracy = torch.zeros(1)
            _, predicted = torch.max(prediction.data, 1)
            total = target.size(0)
            correct = (predicted == target).sum().item()
            accuracy[0] = correct / total

            total_target.append(target.cpu())
            total_predict.append(predicted.cpu())

            print(target)
            print(predicted)

            writer.add_scalar("Downstream_loss/test_step", out_loss,
                              (epoch - 1) * len(downstream_dataloader) + batch_idx)
            total_loss += len(waveform) * out_loss
            writer.add_scalar("Downstream_accuracy/test_step", accuracy * 100,
                              (epoch - 1) * len(downstream_dataloader) + batch_idx)
            total_accuracy += len(waveform) * accuracy

            if batch_idx % 20 == 0:
                large_output, small_output, early_output = trainer.downstream_representations(representations)
                tensorboard.add_latents_heatmap(
                    writer=writer, output=large_output,
                    title="downstream-test-large", desc="test-large",
                    step=(epoch - 1) * len(downstream_dataloader) + batch_idx)
                tensorboard.add_latents_heatmap(
                    writer=writer, output=small_output,
                    title="downstream-test-small", desc="test-small",
                    step=(epoch - 1) * len(downstream_dataloader) + batch_idx)
                tensorboard.add_latents_heatmap(
                    writer=writer, output=early_output,
                    title="downstream-test-early", desc="test-early",
                    step=(epoch - 1) * len(downstream_dataloader) + batch_idx)

    total_loss /= len(downstream_dataloader.dataset)  # average loss
    writer.add_scalar('Downstream_loss/test_epoch', total_loss, (epoch - 1))
    total_accuracy /= len(downstream_dataloader.dataset)  # average acc
    writer.add_scalar('Downstream_accuracy/test_epoch', total_accuracy * 100, (epoch - 1))

    total_target = torch.cat(total_target, dim=0).numpy()
    total_predict = torch.cat(total_predict, dim=0).numpy()
    tensorboard.add_confusion_matrix(writer=writer, title="downstream-test-confusion_matrix", desc="test",
                                     step=(epoch-1), label_num=config['downstream_output_dim'],
                                     targets=total_target, predicts=total_predict)

    tensorboard.add_classification_matrix(config=config, epoch=epoch, writer=writer, title="downstream-test-classification_matrix", desc="test",
                                     step=(epoch - 1), label_num=config['downstream_output_dim'],
                                     targets=total_target, predicts=total_predict)

    tensorboard.add_classification_avg_matrix(writer=writer, title="downstream-test-classification_avg_matrix", desc="test",
                                     step=(epoch - 1), label_num=config['downstream_output_dim'],
                                     targets=total_target, predicts=total_predict)



    return total_loss


def test_downstream_transfer(config, downstream_model, downstream_dataloader, downstream_criterion,
                    writer, epoch, label_dict):
    total_loss = 0.0
    total_accuracy = 0.0
    total_target = []
    total_predict = []
    downstream_model.eval()
    with torch.no_grad():
        for batch_idx, (waveform, label) in enumerate(downstream_dataloader):
            target = trainer.make_downstream_target(label, label_dict)

            if config['use_cuda']:
                waveform = waveform.cuda()
                target = target.cuda()

            prediction, representation = downstream_model(waveform)
            out_loss = downstream_criterion(prediction, target)

            accuracy = torch.zeros(1)
            _, predicted = torch.max(prediction.data, 1)
            total = target.size(0)
            correct = (predicted == target).sum().item()
            accuracy[0] = correct / total

            total_target.append(target.cpu())
            total_predict.append(predicted.cpu())

            writer.add_scalar("Downstream_loss/test_step", out_loss,
                              (epoch - 1) * len(downstream_dataloader) + batch_idx)
            total_loss += len(waveform) * out_loss
            writer.add_scalar("Downstream_accuracy/test_step", accuracy * 100,
                              (epoch - 1) * len(downstream_dataloader) + batch_idx)
            total_accuracy += len(waveform) * accuracy

            # if batch_idx % 20 == 0:
            #     large_output, small_output = trainer.downstream_representations(representation)
            #     tensorboard.add_latents_heatmap(
            #         writer=writer, output=large_output,
            #         title="Downstream_latent_space", desc="test_large",
            #         step=(epoch - 1) * len(downstream_dataloader) + batch_idx)
            #     tensorboard.add_latents_heatmap(
            #         writer=writer, output=small_output,
            #         title="Downstream_latent_space", desc="test_small",
            #         step=(epoch - 1) * len(downstream_dataloader) + batch_idx)

    total_loss /= len(downstream_dataloader.dataset)  # average loss
    writer.add_scalar('Downstream_loss/test_epoch', total_loss, (epoch - 1))
    total_accuracy /= len(downstream_dataloader.dataset)  # average acc
    writer.add_scalar('Downstream_accuracy/test_epoch', total_accuracy * 100, (epoch - 1))

    total_target = torch.cat(total_target, dim=0).numpy()
    total_predict = torch.cat(total_predict, dim=0).numpy()
    tensorboard.add_confusion_matrix(writer=writer, title="downstream-test-confusion_matrix", desc="test",
                                     step=(epoch - 1), label_num=config['downstream_output_dim'],
                                     targets=total_target, predicts=total_predict)

    tensorboard.add_classification_matrix(config=config, epoch=epoch, writer=writer,
                                          title="downstream-test-classification_matrix", desc="test",
                                          step=(epoch - 1), label_num=config['downstream_output_dim'],
                                          targets=total_target, predicts=total_predict)

    tensorboard.add_classification_avg_matrix(writer=writer, title="downstream-test-classification_avg_matrix",
                                              desc="test",
                                              step=(epoch - 1), label_num=config['downstream_output_dim'],
                                              targets=total_target, predicts=total_predict)

    return total_loss

