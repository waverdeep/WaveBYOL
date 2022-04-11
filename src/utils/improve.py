import pandas as pd
import src.utils.interface_file_io as file_io
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def main(metadata_path):
    file_list = []
    label = []
    dataset = pd.read_csv(metadata_path)
    dataset = dataset[['emotion', 'path']]
    for data in dataset.values:
        label.append(data[0])
        file_list.append([data[0], '../../dataset/IEMOCAP_full_release/{}'.format(data[1])])

    label = set(label)
    print(label)

    packed = {tick: [] for tick in label}

    for file in file_list:
        file_label = file[0]
        packed[file_label].append(file[1])

    train_dataset = []
    test_dataset = []

    for key, value in packed.items():
        if len(value) > 0:
            train, test = train_test_split(value, test_size=0.25, random_state=777)
            for temp in train:
                train_dataset.append('{} {}'.format(temp, key))
            for temp in test:
                test_dataset.append('{} {}'.format(temp, key))

    file_io.make_list2txt(train_dataset, '../../dataset/iemocap-train.txt')
    file_io.make_list2txt(test_dataset, '../../dataset/iemocap-test.txt')
    file_io.make_list2txt(label, '../../dataset/iemocap-label.txt')








if __name__ == '__main__':
    path = '../../dataset/IEMOCAP_full_release/iemocap_full_dataset.csv'

    main(path)