import torch
import torchaudio


class animal_audio_classifier(torch.nn.Module):
    def __init__(self):
        super(animal_audio_classifier, self).__init__()

        # TODO improve architecture, this is just the simplest i could find.
        # could also just do a CNN on the image
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

    def visualise(self):
        #TODO visualise model architecture
        pass