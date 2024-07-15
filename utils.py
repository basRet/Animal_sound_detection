from matplotlib import pyplot as plt

def visualize_spectrogram(spectrogram):
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