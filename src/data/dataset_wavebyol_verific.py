from torch.utils.data import Dataset
import src.utils.interface_file_io as file_io
import src.utils.interface_audio_io as audio_io
import src.utils.interface_audio_augmentation as audio_augmentation
import src.utils.interface_audio_augmentation_verific as audio_augmentation_verific
import src.data.dataset_wavebyol as dataset_wavebyol
from src.data import dataset as dataset
import random


# 1, 2, 3, 4, 5, 6 -> 이거 다 박고 들어가면 어떨까
class WaveformDatasetByWaveBYOLVerification01(Dataset):
    def __init__(self, file_path, audio_window=20480, sampling_rate=16000, augmentation=None, augmentation_count=5,
                 randomness=True, config=None):
        super(WaveformDatasetByWaveBYOLVerification01, self).__init__()
        self.file_path = file_path
        self.audio_window = audio_window
        self.sampling_rate = sampling_rate
        self.augmentation = augmentation
        self.augmentation_count = augmentation_count
        self.randomness = randomness
        self.file_list = file_io.read_txt2list(self.file_path)
        self.config = config
        self.selectable_audio_augmentation = audio_augmentation_verific.SelectableAudioAugment(
            audio_window=self.audio_window,
            sample_rate=self.sampling_rate,
            noise_path=self.config['noise_source_path'],
            pitch_shift_value=self.config['pitch_shift_value'],
            reverberation_value=self.config['reverberation_value'],
            time_dropout_value=self.config['time_dropout_value'],
            additive_noise_value=self.config['additive_noise_value'],
            clipping_value=self.config['clipping_audio_value'],
            min_audio_speed=config['min_audio_speed'],
            fix_audio_length=config['fix_audio_length']
        )

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

        if len(self.augmentation) != 0:
            waveform01 = self.selectable_audio_augmentation.get_augmented_audio(
                waveform01, random.sample(self.augmentation, self.augmentation_count), self.randomness)
            waveform02 = self.selectable_audio_augmentation.get_augmented_audio(
                waveform02, random.sample(self.augmentation, self.augmentation_count), self.randomness)
        return waveform01, waveform02