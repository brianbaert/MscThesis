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
