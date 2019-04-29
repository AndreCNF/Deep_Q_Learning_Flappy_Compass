"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch                            # PyTorch to create and apply deep learning models
from torch import nn                    # nn for neural network layers
from torch.nn import functional as F    # Module containing several activation functions

class DeepQNetwork(nn.Module):
    def __init__(self, image_height, image_width, n_inputs=4, conv_dim=[32, 64, 64], 
                 conv_kernel_sizes=[8, 4, 3], conv_strides=[4, 2, 1], fc_dim=[512], n_outputs=2):
        super(DeepQNetwork, self).__init__()

        # Model hyperparameters
        self.image_height = image_height                    # Input image height
        self.image_width = image_width                      # Input image width
        self.n_hidden_conv = len(conv_dim)                  # Number of hidden convolutional layers
        self.n_hidden_fc = len(fc_dim)                      # Number of hidden fully connected layers
        self.n_inputs = n_inputs                            # Number of inputs i.e. input image depth (for instance, in color images, it would be 3 for the RGB channels)
        self.conv_dim = conv_dim                            # List of the number of filters (or filter depth) of each convolutional layer
        self.conv_kernel_sizes = conv_kernel_sizes          # List of the kernel size (or convolutional matrix dimension) of each convolutional layer
        self.conv_strides = conv_strides                    # List of the stride used in each convolutional layer
        self.fc_dim = fc_dim                                # List of the output dimension of each hidden fully connected layer (if there are any)
        self.n_outputs = n_outputs                          # Number of outputs of the neural network

        # Function to compute the dimension of the output of a convolutional layer, considering its parameters and the input image
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Convolutional layers
        self.conv_layers = nn.Sequential()
        self.conv_layers.add_module('conv1', nn.Conv2d(self.n_inputs, self.conv_dim[0],
                                                       kernel_size=self.conv_kernel_sizes[0], stride=self.conv_strides[0]))
        self.conv_layers.add_module('relu1', nn.ReLU())

        # Calculate the output dimension
        conv_output_dim_height = conv2d_size_out(self.image_height, 
                                                 kernel_size=self.conv_kernel_sizes[0], 
                                                 stride=self.conv_strides[0])
        conv_output_dim_width = conv2d_size_out(self.image_width,
                                                kernel_size=self.conv_kernel_sizes[0],
                                                stride=self.conv_strides[0])

        for layer_num in range(1, self.n_hidden_conv):
            # Add the convolutional layer to the dictionary
            self.conv_layers.add_module(f'conv{layer_num+1}', nn.Conv2d(self.conv_dim[layer_num-1], self.conv_dim[layer_num],
                                                                        kernel_size=self.conv_kernel_sizes[layer_num], 
                                                                        stride=self.conv_strides[layer_num]))
            self.conv_layers.add_module(f'relu{layer_num+1}', nn.ReLU())

            # Calculate the output dimension
            conv_output_dim_height = conv2d_size_out(conv_output_dim_height,
                                                     kernel_size=self.conv_kernel_sizes[layer_num], 
                                                     stride=self.conv_strides[layer_num])
            conv_output_dim_width = conv2d_size_out(conv_output_dim_width,
                                                    kernel_size=self.conv_kernel_sizes[layer_num],
                                                    stride=self.conv_strides[layer_num])
                                                    
        # Fully connected layers
        if self.n_hidden_fc == 0:
            # Just one fully connected layer
            self.fc_layers = nn.Linear(conv_output_dim_height * conv_output_dim_width * self.conv_dim[layer_num], 
                                       self.n_outputs)
        else:
            self.fc_layers = nn.Sequential()

            # First fully connected layer
            layer_num += 1
            self.fc_layers.add_module('fc1', nn.Linear(conv_output_dim_height * conv_output_dim_width * self.conv_dim[layer_num-1],
                                            self.fc_dim[0]))
            self.fc_layers.add_module(f'relu{layer_num+1}', nn.ReLU())

            for fc_layer_num in range(1, self.n_hidden_fc):
                layer_num += 1
                self.fc_layers.add_module(f'fc{fc_layer_num+1}', nn.Linear(self.fc_dim[fc_layer_num-1], self.fc_dim[fc_layer_num]))
                self.fc_layers.add_module(f'relu{layer_num+1}', nn.ReLU())

            if self.n_hidden_fc > 1:
                fc_layer_num += 1
            else:
                fc_layer_num = 1

            # Final fully connected layer
            self.fc_layers.add_module(f'fc{fc_layer_num+1}', nn.Linear(self.fc_dim[fc_layer_num-1], self.n_outputs))

        self._create_weights()                              # Initialize the weights

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)     # Create weights from a uniform distribution between -0.01 and 0.01
                nn.init.constant_(m.bias, 0)                # Make the biases zero valued

    def forward(self, input):
        # Feed the input to the convolutional layers, followed by a ReLU activation
        output = self.conv_layers(input)

        # Flatten the output from the convolutional layers
        output = output.view(output.size(0), -1)

        # Feed the convolutional layers' flattened output to the fully connected layers
        output = self.fc_layers(output)

        return output
