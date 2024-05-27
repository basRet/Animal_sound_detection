import torch
import torchaudio


class animal_audio_classifier(torch.nn.Module):
    def __init__(self):
        super(animal_audio_classifier, self).__init__()

        # TODO improve architecture, this is just the simplest i could find.
        # maybe for the input, i could do fourier series as inputs. (A1, B1, A2, B2) or even complex fourier (C1, C2, C3).
        # THis way, it takes some time to preprocess the dataset but afterwards the information density is much higher
        # perhaps convolutions could help where similair frequency matter -> lots of freq. AROUND 500 Hz matter a lot,
        # whereas just one burst might matter less.
        # hmm, you need some way to look at distinces based on some mathematical relationship that it could find.
        # For example, harmonics.
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