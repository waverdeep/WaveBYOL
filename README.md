# WaveBYOL

Self-Supervised Learning for Audio Representation from Raw Waveform

## Description
We propose WaveBYOL model that can learn general-purpose audio representations directly from raw waveforms based on bootstrap your own latent (BYOL) approach, a Siamese neural network architecture. WaveBYOL does not extract features through a handcrafted manner and the model learns general-purpose audio representations from raw waveforms by itself, so it can be easily applied to various downstream tasks. WaveBYOL's Augmentation Layer is designed to create various views in the time domain of raw waveform, and the Encoding Layer is designed to learn representations by extracting features from views, which are augmented audio waveforms. We assessed the representations learned by WaveBYOL by conducting experiments with seven audio downstream tasks both under linear evaluation and transfer learning. Accuracy, Precision, Recall, and F1-score were observed as performance evaluation metrics of the proposed model, and the Accuracy was compared with the existing models. WaveBYOL outperforms the current state-of-the-art models such as contrastive learning for audio (COLA), BYOL for audio (BYOL-A), self-supervised audio spectrogram transformer (SSAST), and DeLoRes in most downstream tasks. Our implementation and pretrained models are given on GitHub.

## Getting Started

### Datasets
* [FSD50K](https://arxiv.org/abs/2010.00475) - Training Pretext task
* [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) - Downstream task
* [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) - Downstream task
* [SpeechCommandV2](https://arxiv.org/abs/1804.03209) - Downstream task
* [RAVDESS](https://smartlaboratory.org/ravdess/) - Downstream task
* [VoxForge](http://www.voxforge.org/home) - Downsteam task
* [NSynth](https://magenta.tensorflow.org/nsynth) - Downstream task
* [ESC-50](https://github.com/karolpiczak/ESC-50) - Downstream task

### Dependencies

* Linux Ubuntu, Nvidia Docker, Python
* adamp 0.3.0
* scikit-learn 1.0.2
* numpy 1.21.6
* tensorboard 2.8.0
* torch 1.12.0
* torchvision 0.13.0
* torchaudio 0.12.0
* tqdm 4.63.1
* sox 1.4.1
* soundfile 0.10.3
* natsort 8.1.0
* [WavAugment](https://github.com/facebookresearch/WavAugment)

### Pretext Task

1. Make up own your configuration file.  (There is an pretext example in the config folder)
2. You can modify this part at [train.py](https://github.com/waverDeep/ImageBYOL/blob/master/train.py)
3. Here's [pretrained model](https://drive.google.com/file/d/1WiywJO5wNqpDGHDYLEn6mAjPGA623Jsf/view?usp=sharing)
```
...
...

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # input tranning cuda device number


...
...


def main():
    parser = argparse.ArgumentParser(description='waverdeep - WaveBYOL')
    parser.add_argument("--configuration", required=False,
                        default='./config/write down your configuration file name')
                        
...
...

```
3. And then, start pretext task training!
```
python train.py
```


### Downstream Task

Currently, only transfer learning is implemented in this project.
1. Make up own your configuration file.  (There is an transfer learning example in the config folder)
2. You can modify this part at [train.py](https://github.com/waverDeep/ImageBYOL/blob/master/train.py)
```
...
...

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # input tranning cuda device number


...
...


def main():
    parser = argparse.ArgumentParser(description='waverdeep - WaveBYOL')
    parser.add_argument("--configuration", required=False,
                        default='./config/write down your configuration file name')
                        
...
...
```

3. And then, start pretext task training!

```
python train.py
```


## Authors

[waverDeep](https://github.com/waverDeep)

## Version History

    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* Thanks to [Sungwoo Moon](https://github.com/Moon-sung-woo) 🙏
* [BYOL](https://github.com/lucidrains/byol-pytorch)
* [BYOL-A](https://github.com/nttcslab/byol-a)
* [Spijkervet/contrastive-predictive-coding](https://github.com/Spijkervet/contrastive-predictive-coding)
* [jefflai108/Contrastive-Predictive-Coding-PyTorch](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch)
* [WavAugment](https://github.com/facebookresearch/WavAugment)
