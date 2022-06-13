import src.utils.interface_file_io as file_io


def merge_txtfiles(filelist, out_filename):
    pack = []
    for file in filelist:
        pack += file_io.read_txt2list(file)

    file_io.make_list2txt(pack, out_filename)


if __name__ == '__main__':
    train_files = [
        '../../dataset/FSD50K.dev_audio_16k.txt',
        '../../dataset/esc-50-new-train.txt',
        '../../dataset/kspon-train.txt',
        '../../dataset/librispeech-train.txt',
        '../../dataset/nsynth-train.txt',
        '../../dataset/ravdess-new-train.txt',
        '../../dataset/speech_commands-new-train.txt',
        '../../dataset/urbansound-new-train.txt',
        '../../dataset/voxceleb01-SI-train.txt',
        '../../dataset/voxforge-new-train.txt',
        '../../dataset/zeroth-train.txt'
    ]

    eval_files = [
        '../../dataset/FSD50K.eval_audio_16k.txt',
        '../../dataset/esc-50-new-test.txt',
        '../../dataset/kspon-test.txt',
        '../../dataset/librispeech-test.txt',
        '../../dataset/nsynth-test.txt',
        '../../dataset/ravdess-new-test.txt',
        '../../dataset/speech_commands-new-test.txt',
        '../../dataset/urbansound-new-test.txt',
        '../../dataset/voxceleb01-SI-test.txt',
        '../../dataset/voxforge-new-test.txt',
        '../../dataset/zeroth-test.txt'
    ]

    merge_txtfiles(train_files, '../../dataset/totaly-train.txt')
    merge_txtfiles(eval_files, '../../dataset/totaly-test.txt')
