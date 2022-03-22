import src.utils.interface_file_io as file_io
import src.utils.interface_audio_io as audio_io


def main():
    dataset_path = '../../dataset/ravdess/'
    file_list = file_io.get_all_file_path(dataset_path, 'wav')

    total_length = 0
    for file in file_list:
        data, sr = audio_io.audio_loader(file)
        total_length += data.size(1)
    print(total_length / len(file_list))
    print(total_length / len(file_list) / 16000)

if __name__ == '__main__':
    main()
