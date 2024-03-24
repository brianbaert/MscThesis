import torch
import torch.nn as nn
import torch.nn.functional as F

#Bahaadini model
class SingleViewModel(nn.Module):
    def __init__(self):
        super(SingleViewModel, self).__init__()

        # First Convolutional Layer with Regularization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(32)

        # Second Convolutional Layer with Regularization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(64)

        # Fully Connected Layer
        self.fc1 = nn.Linear(64*35*42, 256)

        # Output Layer
        self.fc2 = nn.Linear(256, 22)

    def forward(self, x):
        # Applying first conv layer followed by maxpool and dropout
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.5)

        # Applying second conv layer followed by maxpool and dropout
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.5)

        # Flattening the tensor output before feeding it to fully connected layers
        x = x.view(-1, self.num_flat_features(x))

        # Applying fully connected layer followed by dropout
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.05)

        # Output layer with softmax activation function
        out = self.fc2(x)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class BaselineGrayscaleNet_resnet18(nn.Module):
    def __init__(self):
        # Call the class constructor
        super(BaselineGrayscaleNet_resnet18, self).__init__()
        # Initialize a pretrained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        # Replace the first convolutional layer to accept grayscale images (1 channel instead of 3)
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Unfreeze all parameters in the model for training
        for param in self.resnet.parameters():
          param.requires_grad = True  #Unfreeze all parameters
        # Get the number of features (inputs) in the last layer of the model
        num_features_in = self.resnet.fc.in_features
        # Replace the last layer with a new linear layer that matches the number of classes in the dataset, 22
        self.resnet.fc = nn.Linear(num_features_in, 22)

    def forward(self, x):
        # Forward pass: compute the output of the model by passing the input through the model
        x = self.resnet(x)
        # Return the model's output
        return x

class BaselineColorNet_resnet18(nn.Module):
    def __init__(self):
        super(BaselineColorNet_resnet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
          param.requires_grad = True  #Unfreeze all parameters
        num_features_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features_in, 120)
        self.fc2 = nn.Linear(120, 84)
        # 22 classes in GravitySpy dataset
        self.fc3 = nn.Linear(84, 22)
        self.dropout = nn.Dropout(p=0.3)
        self.bn = nn.BatchNorm1d(120)

    def forward(self, x):
        x = self.resnet(x)
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
