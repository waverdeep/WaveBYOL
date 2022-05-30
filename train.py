import argparse
import os
import torch
import json
import numpy as np
import src.utils.interface_train_tool as train_tool
import src.trainers.trainer as trainer
import src.trainers.tester as tester
import src.utils.interface_tensorboard as tensorboard
import src.data.dataset as dataset
import src.models.model as model
import src.optimizers.optimizer as optimizer
import src.optimizers.loss as loss
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    parser = argparse.ArgumentParser(description=' WaveBYOL')
    parser.add_argument("--configuration", required=False,
                        default='./config_FT30/FT10-pretext-WaveBYOL-H2-Kspon-AdamP-20480.json')
    args = parser.parse_args()
    now = train_tool.setup_timestamp()

    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    if 'pretext' in config['train_type']:
        print(">>> Train Pretext - WaveBYOL <<<")
    elif config['train_type'] == 'downstream':
        print(">>> Train Downstream - WaveBYOL <<<")
    print(">> Use GPU: ", torch.cuda.is_available())
    print(">> Config")
    print(config)

    print(">> load dataset ...")
    # setup train/test dataloader
    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, _ = dataset.get_dataloader(config=config, mode='test')

    print("train_loader: {}".format(len(train_loader)))
    print("test_loader: {}".format(len(test_loader)))

    downstream_model = None
    print(">> load pretext model ...")
    pretext_model = model.load_model(config=config, model_name=config["pretext_model_name"],
                                     checkpoint_path=config['pretext_checkpoint'])

    pretext_model_params = sum(p.numel() for p in pretext_model.parameters() if p.requires_grad)
    print("model parameters: {}".format(pretext_model_params))
    print("{}".format(pretext_model))

    if 'downstream_transfer' in config['train_type']:
        print(">> load downstream transfer learning model ...")
        downstream_model = model.load_model(config=config, model_name=config['downstream_model_name'],
                                            checkpoint_path=config['downstream_checkpoint'], pretext_model=pretext_model)

        downstream_model_params = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
        print("model parameters: {}".format(downstream_model_params))
        print("{}".format(downstream_model_params))

        label_dict = train_dataset.label_dict
        print(">> label count: {}".format(len(label_dict.keys())))
    elif 'downstream' in config['train_type']:
        print(">> load downstream model ...")
        downstream_model = model.load_model(config=config, model_name=config['downstream_model_name'],
                                            checkpoint_path=config['downstream_checkpoint'])

        downstream_model_params = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
        print("model parameters: {}".format(downstream_model_params))
        print("{}".format(downstream_model_params))

        label_dict = train_dataset.label_dict
        print(">> label count: {}".format(len(label_dict.keys())))


    print(">> load optimizer ...")
    model_optimizer = None
    if 'pretext' in config['train_type']:
        model_optimizer = optimizer.get_optimizer(pretext_model.parameters(), config)
    elif 'downstream' in config['train_type']:
        print(">> load downstream optimizer")
        model_optimizer = optimizer.get_optimizer(downstream_model.parameters(), config)

    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        if 'downstream' in config['train_type']:
            downstream_model = downstream_model.cuda()

    print(">> set tensorboard ...")
    writer = tensorboard.set_tensorboard_writer("{}-{}".format(config['tensorboard_writer_name'], now))

    print(">> start train/test ...")
    best_loss = None
    early_stop = 0
    epoch = config['epoch']
    for count in range(epoch):
        count = count + 1
        print(">> start train ... [ {}/{} epoch - {} iter ]".format(count, epoch, len(train_loader)))
        if 'pretext' in config['train_type']:
            train_loss = trainer.train_pretext(
                config=config, pretext_model=pretext_model, pretext_dataloader=train_loader,
                pretext_optimizer=model_optimizer, writer=writer, epoch=count)
        elif 'downstream_transfer' in config['train_type']:
            train_loss = trainer.train_downstream_transfer(
                config=config, downstream_model=downstream_model,
                downstream_dataloader=train_loader, downstream_criterion=loss.set_criterion(config['loss_function']),
                downstream_optimizer=model_optimizer, writer=writer, epoch=count, label_dict=label_dict)
        elif 'downstream' in config['train_type']:
            train_loss = trainer.train_downstream(
                config=config, pretext_model=pretext_model, downstream_model=downstream_model,
                downstream_dataloader=train_loader, downstream_criterion=loss.set_criterion(config['loss_function']),
                downstream_optimizer=model_optimizer, writer=writer, epoch=count, label_dict=label_dict)

        print(">> start test  ... [ {}/{} epoch - {} iter ]".format(count, epoch, len(test_loader)))
        if 'pretext' in config['train_type']:
            test_loss = tester.test_pretext(
                config=config, pretext_model=pretext_model,
                pretext_dataloader=test_loader, writer=writer, epoch=count)
        elif 'downstream_transfer' in config['train_type']:
            test_loss = tester.test_downstream_transfer(
                config=config, downstream_model=downstream_model,
                downstream_dataloader=test_loader, downstream_criterion=loss.set_criterion(config['loss_function']),
                writer=writer, epoch=count, label_dict=label_dict)
        elif 'downstream' in config['train_type']:
            test_loss = tester.test_downstream(
                config=config, pretext_model=pretext_model, downstream_model=downstream_model,
                downstream_dataloader=test_loader, downstream_criterion=loss.set_criterion(config['loss_function']),
                writer=writer, epoch=count, label_dict=label_dict)

        if best_loss is None:
            best_loss = test_loss
        elif test_loss < best_loss:
            best_loss = test_loss
            best_epoch = count
            early_stop = 0


            if downstream_model is not None:
                train_tool.save_checkpoint(config=config, model=downstream_model, optimizer=model_optimizer,
                                           loss=test_loss, epoch=best_epoch, mode="best-ds",
                                           date='{}'.format(now))

                print("save downstream checkpoint")
            train_tool.save_checkpoint(config=config, model=pretext_model, optimizer=model_optimizer,
                                       loss=test_loss, epoch=best_epoch, mode="best",
                                       date='{}'.format(now))
            print("save pretext checkpoint")

            print("save checkpoint at {} epoch...".format(count))
        else:
            if downstream_model is not None:
                if test_loss > best_loss:
                    early_stop += 1
                    print(">> count early stop: {}".format(early_stop))

        if early_stop > 9:
            print(">> early stopped... ")
            exit(0)

    tensorboard.close_tensorboard_writer(writer)


if __name__ == '__main__':
    main()


