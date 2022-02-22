import argparse
import os
import torch
import json
import src.utils.interface_train_tool as train_tool
import src.utils.interface_tensorboard as tensorboard
import src.data.dataset as dataset
import src.models.model as model
import src.optimizers.optimizer as optimizer


def main():
    parser = argparse.ArgumentParser(description='waverdeep - WaveBYOL')
    parser.add_argument("--configuration", required=False,
                        default='./config/config_pretext_baseline_FSD50K_T01.json')
    args = parser.parse_args()
    now = train_tool.setup_timestamp()

    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    print(">>> Train Pretext - WaveBYOL <<<")
    print("Use GPU: ", torch.cuda.is_available())
    print(config)

    print("load dataset...")
    # setup train/test dataloader
    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, _ = dataset.get_dataloader(config=config, mode='test')

    print("load model...")
    pretext_model = model.load_model(config, model_name=config["pretext_model_name"])
    print("model structure...")
    model_params = sum(p.numel() for p in pretext_model.parameters() if p.requires_grad)
    print("model parameters: {}".format(model_params))
    print("{}".format(pretext_model))

    if config['use_cuda']:
        pretext_model = pretext_model.cuda()

    print("set optimizer...")
    pretext_optimizer = optimizer.get_optimizer(model_parameter=pretext_model.parameters(), config=config)

    print("set tensorboard...")
    writer = tensorboard.set_tensorboard_writer("{}-{}".format(config['tensorboard_writer_name'], now))

    print("start train/test...")
    best_loss = None
    epoch = config['epoch']
    for num_of_epoch in range(epoch):
        num_of_epoch = num_of_epoch + 1
        print("start train ... [ {}/{} epoch - {} iter ]".format(num_of_epoch, epoch, len(train_loader)))
        train_loss = train(
            config=config,
            writer=writer,
            epoch=num_of_epoch,
            pretext_model=pretext_model,
            data_loader=train_loader,
            pretext_optimizer=pretext_optimizer
        )
        print("start test  ... [ {}/{} epoch - {} iter ]".format(num_of_epoch, epoch, len(test_loader)))
        test_loss = test(
            config=config,
            writer=writer,
            epoch=num_of_epoch,
            pretext_model=pretext_model,
            data_loader=test_loader
        )

        if best_loss is None:
            best_loss = test_loss
        elif test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            train_tool.save_checkpoint(config=config, model=model, optimizer=optimizer,
                                       loss=test_loss, epoch=best_epoch, mode="best",
                                       date='{}'.format(now))
            print("save checkpoint at {} epoch...".format(num_of_epoch))

    tensorboard.close_tensorboard_writer(writer)


def train(config, writer, epoch, pretext_model, data_loader, pretext_optimizer):
    pretext_model.train()
    pretext_model.update_target_weight()
    total_loss = 0.0
    for batch_idx, (waveform01, waveform02) in enumerate(data_loader):
        if config['use_cuda']:
            data01 = waveform01.cuda()
            data02 = waveform02.cuda()
        out_loss, representations = pretext_model(data01, data02)
        pretext_model.zero_grad()
        out_loss.backward()
        pretext_optimizer.step()
        writer.add_scalar('Loss/train_step', out_loss, (epoch - 1) * len(data_loader) + batch_idx)
        total_loss += len(data01) * out_loss

    total_loss /= len(data_loader.dataset)  # average loss
    writer.add_scalar('Loss/train', total_loss, (epoch - 1))
    return total_loss


def test(config, writer, epoch, pretext_model, data_loader):
    pretext_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (waveform01, waveform02) in enumerate(data_loader):
            if config['use_cuda']:
                data01 = waveform01.cuda()
                data02 = waveform02.cuda()
            out_loss, representations = pretext_model(data01, data02)
            writer.add_scalar('Loss/test_step', out_loss, (epoch - 1) * len(data_loader) + batch_idx)
            total_loss += len(data01) * out_loss

        total_loss /= len(data_loader.dataset)  # average loss
        writer.add_scalar('Loss/test', total_loss, (epoch - 1))
    return total_loss


if __name__ == '__main__':
    main()