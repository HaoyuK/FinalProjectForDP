import torch.nn as nn

def calculate_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1

def build_custom_cnn(in_channels, img_size, num_conv_layers=3, num_filters=[32, 64, 128], kernel_size=3, linear_dim=512, num_classes=100):
    layers = []
    current_size = img_size
    
    for i in range(num_conv_layers):
        layers.append(nn.Conv2d(in_channels, num_filters[i], kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_filters[i]))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Update input channels for next layer
        in_channels = num_filters[i]
        
        # Calculate the output size after this layer
        current_size = calculate_output_size(current_size, kernel_size, 1, 1)  # Conv layer
        current_size = calculate_output_size(current_size, 2, 2, 0)           # MaxPool layer
    
    # Calculate the input size for the first fully connected layer
    linear_input_size = num_filters[-1] * current_size * current_size
    
    layers.append(nn.Flatten())
    layers.append(nn.Linear(linear_input_size, linear_dim))
    layers.append(nn.ReLU())
    # layers.append(nn.Linear(linear_dim*2, linear_dim))
    # layers.append(nn.ReLU())
    layers.append(nn.Linear(linear_dim, num_classes))
    
    return nn.Sequential(*layers)

