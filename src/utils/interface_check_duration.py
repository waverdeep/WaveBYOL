import src.utils.interface_file_io as file_io
import src.utils.interface_audio_io as audio_io
from tqdm import tqdm


def check_duration():
    dataset_path = '../../dataset/kspon'
    file_list = file_io.get_all_file_path(dataset_path, 'wav')
    print(dataset_path)
    print(len(file_list))

    total_length = 0
    for file in tqdm(file_list):
        data, sr = audio_io.audio_loader(file)
        total_length += data.size(1)
    print(total_length / len(file_list))
    print(total_length / len(file_list) / 16000)
    print(sr)

if __name__ == '__main__':
    check_duration()
