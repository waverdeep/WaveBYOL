from torch.utils.data import Dataset
import src.utils.interface_file_io as file_io
import src.utils.interface_audio_io as audio_io
import src.utils.interface_audio_augmentation as audio_augmentation
from src.data import dataset as dataset
import torch.nn.functional as F
import random


class WaveformDatasetByWaveBYOL(Dataset):
    def __init__(self, file_path, audio_window=20480, sampling_rate=16000, augmentation=[1, 2, 3, 4, 5, 6],
                 augmentation_count=5):
        super(WaveformDatasetByWaveBYOL, self).__init__()
        self.file_path = file_path
        self.audio_window = audio_window
        self.sampling_rate = sampling_rate
        self.augmentation = augmentation
        self.augmentation_count = augmentation_count
        self.file_list = file_io.read_txt2list(self.file_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_file = dataset.get_audio_filename_path_with_index(self.file_list, index)
        waveform01 = audio_io.audio_adjust_length(
            dataset.load_waveform(audio_file, self.sampling_rate),
            self.audio_window, fit=False)
        waveform02 = audio_io.audio_adjust_length(
            dataset.load_waveform(audio_file, self.sampling_rate),
            self.audio_window, fit=False)

        pick = dataset.get_random_start_point(waveform01.shape[1], self.audio_window)
        waveform01 = audio_io.random_cutoff(waveform01, self.audio_window, pick)
        waveform02 = audio_io.random_cutoff(waveform02, self.audio_window, pick)
        level01 = random.sample(self.augmentation, random.randint(1, self.augmentation_count))
        level01.append(5)
        level02 = random.sample(self.augmentation, random.randint(1, self.augmentation_count))
        level02.append(5)
        if len(self.augmentation) != 0:
            waveform01 = audio_augmentation.audio_augmentation_pipeline(
                waveform01, self.sampling_rate, self.audio_window,
                level01,
                fix_audio_length=True)
            waveform02 = audio_augmentation.audio_augmentation_pipeline(
                waveform02, self.sampling_rate, self.audio_window,
                level02,
                fix_audio_length=True)

        waveform01 = F.normalize(waveform01, dim=-1, p=4)
        waveform02 = F.normalize(waveform02, dim=-1, p=4)

        return waveform01, waveform02
