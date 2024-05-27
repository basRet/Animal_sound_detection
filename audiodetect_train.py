import torch
import torchaudio
import audiodetect_model
import audiodetect_dataset
from matplotlib import pyplot as plt
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
    return figure, axes

if __name__ == '__main__':
    dataset = audiodetect_dataset(root_dir="Animal-Sound-Dataset")
    model = audiodetect_model()

    # train model with no hyperparameter tuning, just the defaults
    # perhaps plot development of accuracy vs epochs during training
    model.train(dataset)

    # evaluate accuracy
    print(model.evaluate())