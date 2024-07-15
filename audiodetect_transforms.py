import torch
import torchaudio
import random

class SpectogramCreation:
    def __init__(self, max_length, decibelConversion = True, fft_bins=800):
        self.decibelConversion = decibelConversion
        self.max_length = max_length
        self.spec_transform = torchaudio.transforms.Spectrogram(fft_bins)
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def pad_spectrogram_right(self, spectrogram):
        _, _, width = spectrogram.shape
        if width < self.max_length:
            padding = self.max_length - width
            # Pad the spectrogram to the right
            spectrogram = torch.nn.functional.pad(spectrogram, (0, padding))
        return spectrogram

    def pad_spectrogram_random(self, spectrogram):
        # Determine the required padding to match the maximum length
        _, _, width = spectrogram.shape
        if self.max_length and width < self.max_length:
            total_padding = self.max_length - width
            left_padding = random.randint(0, total_padding)
            right_padding = total_padding - left_padding
            # Pad the spectrogram on the left and right
            spectrogram = torch.nn.functional.pad(spectrogram, (left_padding, right_padding))
        return spectrogram

    def make_spect(self, waveform, sample_rate):
        if not torch.is_tensor(waveform):
            waveform = torch.tensor(waveform)
        spectrogram = self.spec_transform(waveform)
        if self.decibelConversion:
            spectrogram = self.db_transform(spectrogram)
        spectrogram = self.pad_spectrogram_right(spectrogram)
        return spectrogram

    def __call__(self, sample):
        audio = sample['sample']
        sample_rate = sample['sample_rate']
        label = sample['label']
        image = self.make_spect(sample["sample"], sample["sample_rate"])
        new_sample = {'sample': image, 'sample_rate': sample_rate, 'label': label}
        return new_sample
