import src.utils.interface_file_io as file_io
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def extract_label(filename, name):
    select = None
    if name == 'esc':
        temp = filename.split('/')[-1]
        temp = temp.split('.')[0]
        temp = temp.split('-')[-1]
        select = temp
    elif name == 'ravdess':
        temp = filename.split('/')
        temp = temp[6]
        select = temp.split('-')[2]
    elif name == 'speech_commands':
        temp = filename.split('/')
        select = temp[4]
    elif name =='urbansound':
        filename = filename.split('/')[6]
        select = filename.split('-')[1]
    elif name =='voxforge':
        filename = filename.split('/')[5]
        select = filename#.split('-')[1]
    return select


def main(dataset_path, dataset_name, label_set):
    file_list = file_io.get_all_file_path(dataset_path, 'wav')
    print(file_list[0])
    labels = file_io.read_txt2list(label_set)
    dataset = {tick : [] for tick in labels}

    for file in tqdm(file_list):
        file_label = extract_label(file, dataset_name)
        dataset[file_label].append(file)

    print(len(dataset))
    train_dataset = []
    test_dataset = []

    for key, value in dataset.items():
        if len(value) > 10:
            train, test = train_test_split(value, test_size=0.25, random_state=777)
            train_dataset += train
            test_dataset += test
    print(len(train_dataset))
    print(len(test_dataset))
    print(test_dataset[:10])

    file_io.make_list2txt(train_dataset, '../../dataset/voxforge-new-train.txt')
    file_io.make_list2txt(test_dataset, '../../dataset/voxforge-new-test.txt')


if __name__ == '__main__':
    name = 'voxforge'
    label = '../../dataset/voxforge-label.txt'
    path = '../../dataset/voxforge'
    main(dataset_path=path, dataset_name=name, label_set=label)
    #
    # t1 = file_io.read_txt2list('../../dataset/ravdess-song-new-train.txt')
    # t2 = file_io.read_txt2list('../../dataset/ravdess-speech-new-train.txt')
    # out = t1 + t2
    # file_io.make_list2txt(out,'../../dataset/ravdess-new-train.txt' )
    #
    # t1 = file_io.read_txt2list('../../dataset/ravdess-song-new-test.txt')
    # t2 = file_io.read_txt2list('../../dataset/ravdess-speech-new-test.txt')
    # out = t1 + t2
    # file_io.make_list2txt(out, '../../dataset/ravdess-new-test.txt')