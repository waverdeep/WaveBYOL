import src.data.dataset_wavebyol as dataset_wavebyol
import src.data.dataset_urbansound as dataset_urbansound
import src.data.dataset_voxceleb as dataset_voxceleb
import src.data.dataset_nsynth as dataset_nsynth
import src.data.dataset_speech_command as dataset_speech_command
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
    elif dataset_type == 'WaveformDatasetByWaveBYOLTypeA':
        dataset = dataset_wavebyol.WaveformDatasetByWaveBYOLTypeA(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sampling_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count']
        )
    elif dataset_type == 'UrbanSound8KWaveformDatasetByWaveBYOLTypeA':
        dataset = dataset_urbansound.UrbanSound8KWaveformDatasetByWaveBYOLTypeA(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sampling_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count']
        )
    elif dataset_type == 'VoxCelebWaveformDatasetByWaveBYOLTypeA':
        dataset = dataset_voxceleb.VoxCelebWaveformDatasetByWaveBYOLTypeA(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sampling_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count']
        )
    elif dataset_type == 'NsynthWaveformDatasetByWaveBYOLTypeA':
        dataset = dataset_nsynth.NsynthWaveformDatasetByWaveBYOLTypeA(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sampling_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count']
        )
    elif dataset_type == 'SpeechCommandWaveformDatasetByWaveBYOLTypeA':
        dataset = dataset_speech_command.SpeechCommandWaveformDatasetByWaveBYOLTypeA(
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
