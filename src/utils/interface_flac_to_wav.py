import src.utils.interface_file_io as file_io
import src.utils.interface_audio_resample as audio_resample
import src.utils.interface_multiprocessing as multiprocessing


def flac_to_wav(directory_path):
    file_list = file_io.get_all_file_path(directory_path, 'pcm')
    file_list = file_io.list_divider(48, file_list)
    processes = multiprocessing.setup_multiproceesing(audio_resample.get_pcm_to_convert_wav_2, file_list)
    multiprocessing.start_multiprocessing(processes)


if __name__ == '__main__':
    flac_to_wav(directory_path='../../dataset/kspon')






