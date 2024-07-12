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
    #TODO instead of only visualizing them using MEL spectograms,
    # actually create mel spectograms since this is more understandable for the model too.
    #TODO add padding in creating the spectograms already. This is because we need this as input for the cnn

    def __init__(self, root_dir, transform=None, possible_audio_formats: List[str] = [".wav", ".mp3", ".FLAC"]):
        '''
        create dataset using the .wav files
        :param root_dir: directory with the folders with audio files
        :param transform: optional transform function to be applied to sound files when creating dataset
        '''
        self.possible_audio_formats = possible_audio_formats
        self.length_dataset = None
        self.root_dir = root_dir
        self.transform = transform
        self.audio_data = self.get_files_dataframe()
        self.spec_transform = torchaudio.transforms.Spectrogram(n_fft=800)
        self.length_dataset = len(self.audio_data)

    def __len__(self):
        return self.length_dataset

    def make_spect(self, waveform, sample_rate):
        if not torch.is_tensor(waveform):
            waveform = torch.tensor(waveform)
        spectrogram = self.spec_transform(waveform)
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        return spectrogram

    def __getitem__(self, index):
        # do some transformations later when it's somewhat working
        if torch.is_tensor(index):
            index = index.tolist()  # necessary, since sometimes getitem is called with tensors to improve proecssing times

        waveform, sample_rate = torchaudio.load(self.audio_data["path"][index])
        image = self.make_spect(waveform, sample_rate)
        label = self.audio_data["label"][index]
        sample = {'spect': image, 'sample_rate': sample_rate, 'label': label}

        if self.transform:
            sample = self.transform(image)

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
                        data.append({"path": file_path, "label": os.path.split(animal_folder_path)[1]})
        return pd.DataFrame(data)
    def get_filename(self, index):
        '''
        :param index:
        :return: filename of this index
        '''
        file_path = self.audio_data["path"][index]
        last_dir_in_path = os.path.split(file_path)[1]
        return {last_dir_in_path}
    def get_max_item_len(self):
        import heapq

        print(f"len: {self.__len__()}")
        incorfilelist = list()
        max_lengths = []  # Using a heap to keep track of the top 5 longest files
        average_length = 0

        for i in range(len(self)):
            item = self.__getitem__(i)["spect"]
            length = item.shape[2]

            # Update the max_lengths heap
            if len(max_lengths) < 5:
                heapq.heappush(max_lengths, (length, self.get_full_file_path(i)))
            else:
                heapq.heappushpop(max_lengths, (length, self.get_full_file_path(i)))

            average_length += length / self.length_dataset

        max_lengths.sort(reverse=True, key=lambda x: x[0])
        print(f"max length: {max_lengths[0][0]}, average length: {average_length}")
        print("5 longest files:")
        for length, filepath in max_lengths:
            print(f"{filepath}: {length}")
    def average_class_length(self):
        class_lengths = {}

        for i in range(len(self)):
            label = self.audio_data["label"][i]
            item = self.__getitem__(i)["spect"]
            length = item.shape[2]

            if label not in class_lengths:
                class_lengths[label] = []

            class_lengths[label].append(length)

        average_lengths = {label: np.mean(lengths) for label, lengths in class_lengths.items()}

        for label, avg_length in average_lengths.items():
            print(f"Average length for {label}: {avg_length}")

        return average_lengths
    def get_runtime_error_files(self):
        incor_file_list = list()
        for i in range(0, len(self)):
            try:
                item = self.__getitem__(i)["spect"]
            except RuntimeError as e:
                incorrect_file_path = self.get_full_file_path(i)
                incor_file_list.append(incorrect_file_path)
        print(f"error causing files: {incor_file_list}")
        return incor_file_list
    def remove_files(self, list_of_files):
        for item in list_of_files:
            os.remove(item)
    def get_full_file_path(self, index):
        return self.audio_data["path"][index]
    def visualize_spectrogram(self, spectrogram):
        """
        Visualizes a spectrogram tensor.

        Parameters:
        spectrogram (torch.Tensor): The spectrogram tensor of shape (1, height, width)
        """
        if spectrogram.dim() != 3 or spectrogram.size(0) != 1:
            raise ValueError("The spectrogram should be of shape (1, height, width)")

        # Remove the channel dimension
        spectrogram = spectrogram.squeeze(0)

        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram.numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    data = animalSoundsDataset(root_dir="Animal-Sound-Dataset")
    test_spect = data.__getitem__(1)['spect']
    data.visualize_spectrogram(test_spect)
