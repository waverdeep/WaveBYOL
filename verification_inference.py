import src.utils.interface_audio_io as audio_io
import src.utils.interface_audio_resample as audio_resample
import src.models.model as model

import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt


def load_inference_data(parent, compare, duration):
    # read audi0
    parent_waveform, sr = audio_io.audio_loader(parent)
    compare_waveform, sr = audio_io.audio_loader(compare)
    sample_size = int(duration * sr)
    print(sr)
    print(parent_waveform.size())
    print(compare_waveform.size())
    print(sample_size)

    parent_start_time = ((parent_waveform.size(1) - sample_size) / 2) / sr
    parent_waveform = audio_io.cutoff(parent_waveform, sr, parent_start_time, parent_start_time + duration)
    parent_waveform = audio_io.audio_adjust_length(parent_waveform, sample_size, fit=True)
    print(parent_waveform.size())

    compare_start_time = ((compare_waveform.size(1) - sample_size) / 2) / sr
    compare_waveform = audio_io.cutoff(compare_waveform, sr, compare_start_time, compare_start_time + duration)
    compare_waveform = audio_io.audio_adjust_length(compare_waveform, sample_size, fit=True)
    print(compare_waveform.size())

    return parent_waveform, compare_waveform


def load_inference_model(config):
    print(">> load pretext model ...")
    pretext_model = model.load_model(config=config, model_name=config["pretext_model_name"],
                                     checkpoint_path=config['pretext_checkpoint'])
    downstream_model = model.load_model(config=config, model_name=config['downstream_model_name'],
                                        checkpoint_path=config['downstream_checkpoint'], pretext_model=pretext_model)

    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        downstream_model = downstream_model.cuda()

    return downstream_model


def inference(config, downstream_model, parent_waveform, compare_waveform):
    downstream_model.eval()

    waveforms = torch.stack([parent_waveform, compare_waveform], dim=0)
    if config['use_cuda']:
        waveforms = waveforms.cuda()

    with torch.no_grad():
        embedding = downstream_model.get_embedding(waveforms)
    return embedding



def calculate_similarity(parent_prediction, compare_prediction):
    cosine_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    output = cosine_sim(parent_prediction, compare_prediction)
    return output


def main(config, parent, compare, duration):

    with open(config, 'r') as configuration:
        config = json.load(configuration)

    parent_waveform, compare_waveform = load_inference_data(parent, compare, duration)
    downstream_model = load_inference_model(config)
    prediction = inference(config, downstream_model, parent_waveform, compare_waveform)
    if config['use_cuda']:
        prediction = prediction.cpu()
    parent_prediction = prediction[0]#.unsqueeze(0)
    compare_prediction = prediction[1]#.unsqueeze(0)
    print(prediction.size())

    # fig = plt.figure()
    # plt.matshow(parent_prediction[0].numpy())
    # plt.show()
    # plt.close()
    # plt.matshow(compare_prediction[0].numpy())
    # plt.show()
    # plt.close()

    output = calculate_similarity(parent_prediction, compare_prediction)
    print(output.mean())


if __name__ == '__main__':
    parent_data_path = "./dataset_test/test-lee_16000.wav"
    compare_data_path = "./dataset_test/test-u1_16000.wav"
    model_config = "./config_VRFCTN/verification01.json"
    seek_duration = 4
    # audio_resample.resample_audio("./dataset_test/test-mun.wav", original_sr=44100, convert_sr=16000)
    # audio_resample.resample_audio("./dataset_test/test-bae.wav", original_sr=44100, convert_sr=16000)
    # audio_resample.resample_audio("./dataset_test/test-u1.wav", original_sr=44100, convert_sr=16000)
    # audio_resample.resample_audio("./dataset_test/test-u2.wav", original_sr=44100, convert_sr=16000)
    # audio_resample.resample_audio("./dataset_test/test-lee.wav", original_sr=44100, convert_sr=16000)
    main(config=model_config, parent=parent_data_path, compare=compare_data_path, duration=seek_duration)
