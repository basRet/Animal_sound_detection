import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
from typing import List
from matplotlib import pyplot as plt

# import audio
class animalSoundsDataset(Dataset):
    def __init__(self, root_dir, transform = None, possible_audio_formats: List[str] = [".wav", ".mp3", ".FLAC"]):
        '''
        create dataset using the .wav files
        :param root_dir: directory with the folders with audio files
        :param transform: optional transform function to be applied to sound files when creating dataset
        '''
        self.possible_audio_formats = possible_audio_formats
        length_dataset = None
        self.root_dir = root_dir
        self.transform = transform
        self.audio_data = self.get_files_dataframe()

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        # do some transformations later when it's somewhat working
        if torch.is_tensor(index):
            index = index.tolist()  # necessary, since sometimes getitem is called with tensors to improve proecssing times

        audio = torchaudio.load(self.audio_data["path"][index])
        label = self.audio_data["label"][index]
        sample = {'audio': audio, 'label': label}

        if self.transform:
            sample = self.transform(audio)

        return sample

    def get_files_dataframe(self):
        '''
        creates file paths and labels for every data sample, puts in dataframe
        :return:
        '''
        data = []
        for animal_folder in os.listdir(self.root_dir):
            animal_folder_path = os.path.join(self.root_dir, animal_folder)
            if os.path.isdir(animal_folder_path):
                for file_name in os.listdir(animal_folder_path):
                    file_path = os.path.join(animal_folder_path, file_name)
                    if os.path.splitext(file_path)[1] in self.possible_audio_formats:
                        # add ID, path as path, animal folder path as label
                        data.append({"path" : file_path, "label" : os.path.split(animal_folder_path)[1]})
        return pd.DataFrame(data)

    def get_filename(self, index):
        '''
        :param index:
        :return: filename of this index
        '''
        file_path = self.audio_data["path"][index]
        last_dir_in_path = os.path.split(file_path)[1]
        extension = os.path.splitext(file_path)[1]
        return f'{last_dir_in_path}{extension}'

N_FFT = 4096
N_HOP = 4
stft = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=N_HOP,
    power=None,
)

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show()


# for bugfixing, visualise the waves.
if __name__ == '__main__':
    print(torchaudio.list_audio_backends())

    data = animalSoundsDataset(root_dir="Animal-Sound-Dataset")
    index = 761
    sample = data.__getitem__(index)
    filename=data.get_filename(index)
    audio, sample_rate = sample["audio"]
    plot_specgram(audio, sample_rate, title=f"spectrogram of {filename}")

