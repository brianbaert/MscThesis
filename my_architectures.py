import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import List, Optional
import avalanche as avl
import avalanche.models
from avalanche.models import SCRModel, MTSlimResNet18
from avalanche.evaluation import metrics as metrics
from avalanche.benchmarks.scenarios import CLExperience
from avalanche.benchmarks.utils.flat_data import ConstantSequence
from torch.nn.functional import relu, avg_pool2d
from avalanche.models import BaseModel, MultiHeadClassifier, MultiTaskModule, DynamicModule
from torchvision.models.resnet import BasicBlock

class BaselineGrayscaleNet_resnet18(nn.Module):
    def __init__(self, num_classes=10):
        # Call the class constructor
        super(BaselineGrayscaleNet_resnet18, self).__init__()
        # Initialize a pretrained ResNet-18 model
        self.resnet = models.resnet18(weights='DEFAULT')
        # Replace the first convolutional layer to accept grayscale images (1 channel instead of 3)
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Unfreeze all parameters in the model for training
        for param in self.resnet.parameters():
          param.requires_grad = True  #Unfreeze all parameters
        # Get the number of features (inputs) in the last layer of the model
        num_features_in = self.resnet.fc.in_features
        # Replace the last layer with a new linear layer that matches the number of classes in the dataset, 22
        self.resnet.fc = nn.Linear(num_features_in, num_classes)

    def forward(self, x):
        # Forward pass: compute the output of the model by passing the input through the model
        x = self.resnet(x)
        # Return the model's output
        return x

class BaselineColorNet_resnet18(nn.Module):
    def __init__(self, num_classes=10):
        # Call the parent class's constructor
        super(BaselineColorNet_resnet18, self).__init__()
        # Initialize a pretrained ResNet-18 model
        self.resnet = models.resnet18(weights='DEFAULT')
        # Unfreeze all parameters in the model for training
        for param in self.resnet.parameters():
          param.requires_grad = True  
        # Get the number of features in the last layer of the model
        num_features_in = self.resnet.fc.in_features
        # Replace the last layer with a new linear layer
        self.resnet.fc = nn.Linear(num_features_in, 120)
        # Add a second fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # Add a third fully connected layer for the num_classes classes in the GravitySpy dataset
        self.fc3 = nn.Linear(84, num_classes)
        # Add a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=0.3)
        # Add a batch normalization layer
        self.bn = nn.BatchNorm1d(120)

    def forward(self, x):
        # Forward pass: compute the output of the model by passing the input through the model
        x = self.resnet(x)
        # Apply batch normalization
        x = self.bn(x)
        # Apply the ReLU activation function
        x = F.relu(self.fc2(x))
        # Apply dropout
        x = self.dropout(x)
        # Pass the result through the final fully connected layer
        x = self.fc3(x)
        # Return the model's output
        return x
    
class MultiViewColorNet_resnet18(nn.Module):
  def __init__(self, num_classes=10):
    # Call the parent class's constructor
    super(MultiViewColorNet_resnet18, self).__init__()

    # Initialize a pretrained ResNet-18 model with adjusted input size
    self.resnet = models.resnet18(weights='DEFAULT')

    # Access the first convolutional layer (assuming it's named conv1)
    first_conv = self.resnet.conv1

    # Modify the kernel size of the first convolution
    first_conv.kernel_size = (7, 7)

    # Unfreeze all parameters in the model for training
    for param in self.resnet.parameters():
      param.requires_grad = True

    # Get the number of features in the last layer of the model
    num_features_in = self.resnet.fc.in_features

    # Replace the last layer with a new linear layer
    self.resnet.fc = nn.Linear(num_features_in, 120)
    # Add a second fully connected layer
    self.fc2 = nn.Linear(120, 84)
    # Add a third fully connected layer for the num_classes classes in the GravitySpy dataset
    self.fc3 = nn.Linear(84, num_classes)
    # Add a dropout layer to prevent overfitting
    self.dropout = nn.Dropout(p=0.3)
    # Add a batch normalization layer
    self.bn = nn.BatchNorm1d(120)

  def forward(self, x):
    # Forward pass: compute the output of the model by passing the input through the model
    x = self.resnet(x)
    # Apply batch normalization
    x = self.bn(x)
    # Apply the ReLU activation function
    x = F.relu(self.fc2(x))
    # Apply dropout
    x = self.dropout(x)
    # Pass the result through the final fully connected layer
    x = self.fc3(x)
    # Return the model's output
    return x

class FractalDimensionConvNet(nn.Module):
    def __init__(self):
        super(FractalDimensionConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=50, out_channels=128, kernel_size=3, stride=1, padding=1) #adjust from 347 to 50
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64*56, 1024)
        #self.fc1 = nn.Linear(64*5, 1024) #adjust input size for fd statistics
        self.fc2 = nn.Linear(1024,256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256,3)

    def forward(self, x):
        x = nn.functional.selu(self.conv1(x))
        x = self.bn1(x)
        x = nn.functional.selu(self.conv2(x))
        x = self.bn2(x)
        x = nn.functional.selu(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout1(x)
        x = nn.functional.selu(self.conv4(x))
        x = self.bn4(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.selu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MultiModalNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiModalNet, self).__init__()
        self.colorNet = MultiViewColorNet_resnet18(num_classes)
        self.fractalNet = FractalDimensionConvNet()
        # Define a new output layer for the combined features
        self.fc = nn.Linear(6, num_classes)

    def forward(self, x1, x2):
        # Forward pass through each network
        x1 = self.colorNet(x1)
        x2 = self.fractalNet(x2)
        # Concatenate the outputs along dimension 1
        x = torch.cat((x1, x2), dim=1)
        # Pass the combined features through the output layer
        x = self.fc(x)
        return x

