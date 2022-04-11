import os

import soundfile as sf
from tqdm import tqdm
import src.utils.interface_file_io as io

import numpy as np
import librosa
import multiprocessing
import src.utils.interface_multiprocessing as mi


def resample_audio(data_path, original_sr, convert_sr):
    waveform, sampling_rate = librosa.load(data_path, original_sr)
    resample_waveform = librosa.resample(waveform, original_sr, convert_sr)
    new_filename = data_path.replace(".wav", "_16000.wav")
    sf.write(new_filename, resample_waveform, 16000)


def get_audio_list(directory_path_list, new_save_path='../../dataset/FSD50K.dev_audio_16k/', audio_window=20480, file_extension="wav"):
    for directory in directory_path_list:
        file_list = io.get_all_file_path(directory, file_extension)
        for file in tqdm(file_list, desc=directory):
            waveform, sampling_rate = librosa.load(file, 44100)
            resample_waveform = librosa.resample(waveform, 44100, 16000)
            new_filename = file.replace("ravdess_song", "ravdess_song_16k")
            sf.write(new_filename, resample_waveform, 16000)


def get_audio_to_convert_wav(file_list, file_extension="flac"):
    list_size = len(file_list)
    for index, file in enumerate(file_list):
        waveform, sampling_rate = librosa.load(file, 16000)
        new_filename = file.replace(file_extension, 'wav')
        sf.write(new_filename, waveform, sampling_rate, format='WAV', endian='LITTLE', subtype='PCM_16')
        if index % 50 == 0:
            proc = os.getpid()
            print("P{}: {}/{}".format(proc, index, list_size))


def get_pcm_to_convert_wav(file_list, file_extension="pcm"):
    list_size = len(file_list)
    for index, file in enumerate(file_list):
        with open(file, 'rb') as pcm_file:
            buf = pcm_file.read()
            pcm_data = np.frombuffer(buf[0:len(buf)-1], dtype='int16')
            waveform = librosa.util.buf_to_float(pcm_data, 2)
        new_filename = file.replace(file_extension, 'wav')
        sf.write(new_filename, waveform, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
        if index % 50 == 0:
            proc = os.getpid()
            print("P{}: {}/{}".format(proc, index, list_size))



def resampling_audio(file_list):
    list_size = len(file_list)
    for index, file in enumerate(file_list):
        waveform, sampling_rate = librosa.load(file, 44100)
        resample_waveform = librosa.resample(waveform, 44100, 16000)
        new_filename = file.replace("audio", "audio_16k")
        sf.write(new_filename, resample_waveform, 16000)
        if index % 50 == 0:
            proc = os.getpid()
            print("P{}: {}/{}".format(proc, index, list_size))


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def aa():
    directory_path = '../../dataset/kspon/eval_other/'
    file_extension = "pcm"
    file_list = io.get_all_file_path(directory_path, file_extension)
    get_pcm_to_convert_wav(file_list)

def aaa():
    # for i in range(1, 25):
    #     create_folder('../../dataset/ravdess/ravdess_speech_16k/Actor_{}'.format(str(i).zfill(2)))
    directory_path = '../../dataset/UrbanSound8K/audio/'
    file_extension = "wav"
    divide_num = multiprocessing.cpu_count() - 1
    print(divide_num)
    file_list = io.get_all_file_path(directory_path, file_extension)
    print(file_list[:10])
    file_list = io.list_divider(divide_num, file_list)
    print(len(file_list))

    processes = mi.setup_multiproceesing(resampling_audio, data_list=file_list)
    mi.start_multiprocessing(processes)


if __name__ == '__main__':
    aaa()
    # resampling_audio()
    # directory_path = ['../../dataset/FSD50K.dev_audio']
    # get_audio_list(directory_path, '../../dataset/FSD50K.dev_audio_16k/', audio_window=20480, file_extension="wav")

    # directory_path = ['../../dataset/UrbanSound8K/audio']
    # get_audio_list(directory_path, '../../dataset/UrbanSound8K/audio_16k/', audio_window=20480, file_extension="wav")
