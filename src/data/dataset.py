import src.data.dataset_wavebyol as dataset_wavebyol
from torch.utils import data


def get_dataloader(config, mode="train"):
    dataset_type = config['dataset_type']
    dataset = None

    if dataset_type == 'WaveformDatasetByWaveBYOL':
        dataset = dataset_wavebyol.WaveformDatasetByWaveBYOL(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sampling_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count']
        )

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=config['dataset_shuffle'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )

    return dataloader, dataset