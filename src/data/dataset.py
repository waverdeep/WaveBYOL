import src.data.dataset_wavebyol as dataset_wavebyol
import src.data.dataset_downstream as dataset_downstream
import src.utils.interface_audio_io as audio_io
from torch.utils import data
import numpy as np


def get_random_start_point(size, audio_window):
    return np.random.randint(size - audio_window + 1)


def get_audio_filename_path_with_index(file_list, index):
    audio_file = file_list[index]
    return audio_file[4:]


def load_waveform(audio_file, required_sampling_rate):
    waveform, sampling_rate = audio_io.audio_loader(audio_file)
    assert (
            sampling_rate == required_sampling_rate
    ), "sampling rate is not consistent throughout the dataset"
    return waveform


def get_dataloader(config, mode="train"):
    train_type = config['train_type']
    dataset = None

    if 'pretext' in train_type:
        dataset = dataset_wavebyol.WaveformDatasetByWaveBYOL(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sampling_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count']
        )
    elif 'downstream' in train_type:
        dataset = dataset_downstream.WaveformDataset(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sampling_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count'],
            label_file_path=config['label_file_path'],
            metadata=config['metadata'],
            config=config,
            dataset_name=config['dataset_name'],
        )




    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=config['dataset_shuffle'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )

    return dataloader, dataset
