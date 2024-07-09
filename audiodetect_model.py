import torch
import torchaudio

class animal_audio_classifier(torch.nn.Module):
    def __init__(self):
        super(animal_audio_classifier, self).__init__()

        # TODO improve architecture, this is just the simplest i could find.
        # It should be an RNN on the spectograms. Input size is (401, max_duration)
        # couple of RNN layers, then dense layer at the end to classify animal. Perhaps some dense inbetween too
        # output has softmax for classifiying between 0 and 1
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