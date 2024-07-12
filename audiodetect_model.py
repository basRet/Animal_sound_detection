import torch
import torchaudio

class animal_audio_classifier(torch.nn.Module):
    def __init__(self, num_classes=10):
        # define all the layers that can possibly be used in the model
        super(animal_audio_classifier, self).__init__()

        # TODO improve architecture, i found this at
        # TODO Audio Deep Learning Made Simple: Sound Classification, Step-by-Step
        # TODO on medium, by Ketan Doshi on Mar 18, 2021
        # TODO but made it simpler
        #
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm.
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=(8, 8), stride=(2, 2), padding=(2, 2))
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(4)
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = torch.nn.Conv2d(4, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(16)
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Linear Classifier to get features from 16 to number of classes
        self.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = torch.nn.Linear(in_features=16, out_features=num_classes)
        # softmax ensures probabilities as outputs
        self.softmax = torch.nn.Softmax(dim=1)

        # Wrap the Convolutional Blocks
        self.conv = torch.nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        x = self.softmax(x)

        # Final output
        return x

    def visualise(self):
        #TODO visualise model architecture
        pass