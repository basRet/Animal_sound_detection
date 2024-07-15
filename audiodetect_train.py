import torch
import torchaudio
from audiodetect_model import animal_audio_classifier as model
from audiodetect_dataset import animalSoundsDataset as dataset
from utils import visualize_spectrogram
from matplotlib import pyplot as plt
import audiodetect_transforms
MAX_LENGTH = 792000 # look it up before and save it to save processing, can also be got with dataset.get_max_item_len()

if __name__ == '__main__':
    dataset = dataset(root_dir="Animal-Sound-Dataset")
    print(dataset.get_max_item_len())

    spec_creation_transform = audiodetect_transforms.SpectogramCreation(MAX_LENGTH)
    dataset = dataset(root_dir="Animal-Sound-Dataset", transform=spec_creation_transform)
    model = model()


    training_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    validation_loader = None

    for i, item in enumerate(dataset):
        spect = item["sample"]
        label = item["label"]
        sample_rate = item["sample_rate"]
        print(label)
        visualize_spectrogram(spect)
        if i>3:
            break

    # train model with no hyperparameter tuning, just the defaults
    # perhaps plot development of accuracy vs epochs during training
    # model.train(dataset)

    # evaluate accuracy
    # print(model.evaluate())