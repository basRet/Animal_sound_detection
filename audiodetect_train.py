import torch
import torchaudio
from audiodetect_model import animal_audio_classifier as model
from audiodetect_dataset_spectogram import animalSoundsDataset as dataset
from matplotlib import pyplot as plt

if __name__ == '__main__':
    #TODO look up how to do this and then carry it out. This is all pseudocode
    dataset = dataset(root_dir="Animal-Sound-Dataset")
    model = model()

    # train model with no hyperparameter tuning, just the defaults
    # perhaps plot development of accuracy vs epochs during training
    model.train(dataset)

    # evaluate accuracy
    print(model.evaluate())