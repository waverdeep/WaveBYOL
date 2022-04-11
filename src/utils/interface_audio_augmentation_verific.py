import augment
import random
import numpy as np
import torchaudio.sox_effects

import src.utils.interface_file_io as file_io
import src.utils.interface_audio_io as audio_io


class SelectableAudioAugment:
    def __init__(self, audio_window, sample_rate, noise_path="./dataset/musan-total.txt",
                 pitch_shift_value=0, reverberation_value=0, time_dropout_value=0,
                 additive_noise_value=0, clipping_value=0, min_audio_speed=0, fix_audio_length=True):
        super(SelectableAudioAugment, self).__init__()
        self.audio_window = audio_window
        self.sample_rate = sample_rate
        self.source_info = {'rate': self.sample_rate}
        self.target_info = {'rate': self.sample_rate}
        self.noise_path = noise_path
        self.noise_filelist = file_io.read_txt2list(self.noise_path)
        self.pitch_shift_value = pitch_shift_value
        self.reverberation_value = reverberation_value
        self.time_dropout_value = time_dropout_value
        self.additive_noise_value = additive_noise_value
        self.clipping_value = clipping_value
        self.min_audio_speed = min_audio_speed
        self.fix_audio_length = fix_audio_length

    def pitch_shift(self, x):
        random_pitch_shift = np.random.randint(-self.pitch_shift_value, +self.pitch_shift_value)
        combination = augment.EffectChain().pitch(random_pitch_shift).rate(self.sample_rate)
        y = combination.apply(x, src_info=self.source_info, target_info=self.target_info)
        return y

    def reverberation(self, x):
        random_room_size = np.random.randint(0, self.reverberation_value)
        combination = augment.EffectChain().reverb(50, 50, random_room_size).channels(1)
        y = combination.apply(x, src_info=self.source_info, target_info=self.target_info)
        return y

    def time_dropout(self, x):
        combination = augment.EffectChain().time_dropout(max_seconds=self.time_dropout_value)
        y = combination.apply(x, src_info=self.source_info, target_info=self.target_info)
        return y

    def noise_generator(self):
        pick = np.random.randint(len(self.noise_filelist))
        waveform, _ = audio_io.audio_loader(self.noise_filelist[pick][4:])
        waveform = audio_io.audio_adjust_length(waveform, self.audio_window)
        waveform = audio_io.random_cutoff(waveform, self.audio_window)
        return waveform

    def additive_noise(self, x):
        random_snr = np.random.randint(3, self.additive_noise_value)
        combination = augment.EffectChain().additive_noise(self.noise_generator, snr=random_snr)
        y = combination.apply(x, src_info=self.source_info, target_info=self.target_info)
        return y

    def clipping_audio(self, x):
        combination = augment.EffectChain().clip(self.clipping_value)
        y = combination.apply(x, src_info=self.source_info, target_info=self.target_info)
        return y

    def audio_speed(self, x):
        speed = np.random.randint(int(self.min_audio_speed*100), 99) / 100
        effects = [['speed', str(speed)]]
        y, _ = torchaudio.sox_effects.apply_effects_tensor(x, self.sample_rate, effects)
        return y

    def get_augmented_audio(self, x, method_list, randomness=True):
        if randomness:
            random.shuffle(method_list)

        pipeline = []
        for step in method_list:
            if step == 'pitch_shift':
                pipeline.append(self.pitch_shift)
            elif step == 'reverberation':
                pipeline.append(self.reverberation)
            elif step == 'time_dropout':
                pipeline.append(self.time_dropout)
            elif step == 'additive_noise':
                pipeline.append(self.additive_noise)
            elif step == 'clipping_audio':
                pipeline.append(self.clipping_audio)
            elif step == 'audio_speed':
                pipeline.append(self.audio_speed)

        for augment_step in pipeline:
            x = augment_step(x=x)
            if self.fix_audio_length:
                if len(x[0]) != self.audio_window:
                    x = audio_io.audio_adjust_length(x, audio_window=self.audio_window, fit=self.fix_audio_length)
        return x
