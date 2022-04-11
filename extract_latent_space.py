import argparse
import os
import torch
import json
import numpy as np
import src.utils.interface_train_tool as train_tool
import src.utils.interface_audio_io as audio_io
import matplotlib.pyplot as plt
import src.trainers.trainer as trainer
import src.trainers.tester as tester
import src.utils.interface_tensorboard as tensorboard
import src.data.dataset as dataset
import src.models.model as model
import src.optimizers.optimizer as optimizer
import src.optimizers.loss as loss
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main(sample_audio):
    parser = argparse.ArgumentParser(description='waverdeep - WaveBYOL - Feature extractor')
    parser.add_argument("--configuration", required=False,
                        default='./config/T10-urbansound-WaveBYOL-ResNet50-Adam-15200.json')
    args = parser.parse_args()
    now = train_tool.setup_timestamp()

    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    print(">> load pretext model ...")
    pretext_model = model.load_model(config=config, model_name=config["pretext_model_name"],
                                     checkpoint_path=config['pretext_checkpoint'])

    if config['use_cuda']:
        pretext_model = pretext_model.cuda()

    for audio in sample_audio:
        waveform, sr = audio_io.audio_loader(audio)
        waveform = waveform.unsqueeze(0)
        if config['use_cuda']:
            waveform = waveform.cuda()

        with torch.no_grad():
            out_representation = pretext_model.get_representation(waveform)
        out_representation = out_representation.detach()
        print(out_representation.size())
        out_representation = out_representation.cpu().numpy()

        temp = []
        for j in range(1):
            temp.append(out_representation[0][j])
        temp = np.vstack(temp)
        plt.matshow(temp)
        plt.xlabel('time')
        plt.colorbar()
        plt.show()

    for audio in sample_audio:
        waveform, sr = audio_io.audio_loader(audio)
        waveform = waveform.unsqueeze(0)
        if config['use_cuda']:
            waveform = waveform.cuda()

        with torch.no_grad():
            out_representation = pretext_model.get_early_representation(waveform)
        out_representation = out_representation.detach()
        print(out_representation.size())
        out_representation = out_representation.cpu().numpy()

        temp = []
        temp.append(out_representation[0])
        temp = np.vstack(temp)
        plt.matshow(temp)
        plt.xlabel('time')
        plt.ylabel('channel')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    audio_set = [
        './dataset_test/41_0514_301_0_07111_00.wav',
        './dataset_test/41_0514_301_0_07111_01.wav',
        './dataset_test/41_0514_301_0_07111_02.wav',
    ]
    main(audio_set)
