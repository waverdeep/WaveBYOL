from torch.utils.data import Dataset
import src.utils.interface_file_io as file_io
import src.utils.interface_audio_io as audio_io
import src.utils.interface_audio_augmentation as audio_augmentation
from src.data import dataset as dataset
import random
import natsort
import pandas as pd


def get_label_dict(label_list):
    label_dict = {}
    for idx, key in enumerate(label_list):
        label_dict[str(key)] = idx
    return label_dict


def load_audio_with_label(file_list, index, dataset_name=None):
    audio_file = dataset.get_audio_filename_path_with_index(file_list, index)
    if dataset_name == 'Urbansound8K':
        filename = audio_file.split('/')[5]
        label = filename.split('-')[1]
    elif dataset_name == 'Nsynth':
        temp = audio_file.split('/')
        label = temp[4]
        label = label.split('_')[0]
    elif dataset_name == 'Ravdess':
        temp = audio_file.split('/')
        temp = temp[5]
        label = temp.split('-')[2]
    else:
        temp = audio_file.split('/')
        label = temp[4]

    return audio_file, label


class WaveformDataset(Dataset):
    def __init__(self, file_path: str, audio_window=20480, sampling_rate=16000,
                 augmentation=[1, 2, 3, 4, 5, 6], augmentation_count=5,
                 label_file_path=None,
                 metadata=None, config=None, dataset_name=None):
        super(WaveformDataset, self).__init__()
        self.audio_window = audio_window
        self.sampling_rate = sampling_rate
        self.augmentation = augmentation
        self.augmentation_count = augmentation_count
        self.file_list = file_io.read_txt2list(file_path)
        self.metadata = metadata
        self.config = config
        self.dataset_name = dataset_name

        if self.metadata is not None:
            self.metadata = pd.read_csv(metadata)
            self.label_list = natsort.natsorted(list(set(self.metadata['classID'])))
        else:
            self.label_list = natsort.natsorted(file_io.read_txt2list(label_file_path))

        self.label_dict = get_label_dict(self.label_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_file, label = load_audio_with_label(self.file_list, index, self.dataset_name)
        waveform = audio_io.audio_adjust_length(
            dataset.load_waveform(audio_file, self.sampling_rate), self.audio_window, fit=False)

        pick = dataset.get_random_start_point(waveform.shape[1], self.audio_window)
        waveform = audio_io.random_cutoff(waveform, self.audio_window, pick)

        if len(self.augmentation) != 0:
            waveform = audio_augmentation.audio_augmentation_pipeline(
                waveform, self.sampling_rate, self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)

        return waveform, str(label)