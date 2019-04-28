"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch                            # PyTorch to create and apply deep learning models
from torch import nn                    # nn for neural network layers
from torch.nn import functional as F    # Module containing several activation functions

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 2)

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # Feed the input to the convolutional layers, followed by a ReLU activation
        output = F.relu(self.conv1(input))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))

        # Flatten the output from the convolutional layers and pass it to the fully connected layers
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)

        return output
