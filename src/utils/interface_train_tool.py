import argparse
import json
import random
import numpy as np
import os
import torch.cuda
from datetime import datetime
import src.utils.interface_file_io as file_io


def setup_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_timestamp():
    now = datetime.now()
    return "{}_{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)


def setup_config(configuration):
    return file_io.load_json_config(configuration)


def make_target(speaker_id, speaker_dict):
    targets = torch.zeros(len(speaker_id)).long()
    for idx in range(len(speaker_id)):
        targets[idx] = speaker_dict[speaker_id[idx]]
    return targets


def save_checkpoint(config, model, optimizer, loss, epoch, mode="best", date=""):
    if not os.path.exists(os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name'])):
        file_io.make_directory(os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name']))
    base_directory = os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name'])
    if mode == "best":
        file_path = os.path.join(base_directory,
                                 config['checkpoint_file_name'] + "-model-best-{}-epoch-{}.pt".format(date, epoch))
    elif mode == "best-ds":
        file_path = os.path.join(base_directory,
                                 config['checkpoint_file_name'] + "-model-best-ds-{}-epoch-{}.pt".format(date, epoch))
    elif mode == 'step':
        file_path = os.path.join(base_directory,
                                 config['checkpoint_file_name'] + "-model-{}-epoch-{}.pt".format(date, epoch))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "loss": loss}, file_path)