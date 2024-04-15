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


