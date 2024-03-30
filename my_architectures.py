import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import List, Optional
from torch.nn import Module
from avalanche.benchmarks.scenarios import CLExperience
from avalanche.benchmarks.utils.flat_data import ConstantSequence

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
        # Call the parent class's constructor
        super(BaselineColorNet_resnet18, self).__init__()
        # Initialize a pretrained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        # Unfreeze all parameters in the model for training
        for param in self.resnet.parameters():
          param.requires_grad = True  
        # Get the number of features in the last layer of the model
        num_features_in = self.resnet.fc.in_features
        # Replace the last layer with a new linear layer
        self.resnet.fc = nn.Linear(num_features_in, 120)
        # Add a second fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # Add a third fully connected layer for the 22 classes in the GravitySpy dataset
        self.fc3 = nn.Linear(84, 22)
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

################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco, Antonio Carta                                  #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)

class DynamicModule(nn.Module):
    """Dynamic Modules are Avalanche modules that can be incrementally
    expanded to allow architectural modifications (multi-head
    classifiers, progressive networks, ...).

    Compared to pytoch Modules, they provide an additional method,
    `model_adaptation`, which adapts the model given the current experience.
    """

    def __init__(self, auto_adapt=True):
        """
        :param auto_adapt: If True, will be adapted in the recursive adaptation loop
                           else, will be adapted by a module in charge
                           (i.e IncrementalClassifier inside MultiHeadClassifier)
        """
        super().__init__()
        self._auto_adapt = auto_adapt


    def recursive_adaptation(self, experience):
        """
        Calls self.adaptation recursively accross
        the hierarchy of pytorch module childrens
        """
        avalanche_model_adaptation(self, experience)

    def adaptation(self, experience: CLExperience):
        """Adapt the module (freeze units, add units...) using the current
        data. Optimizers must be updated after the model adaptation.

        Avalanche strategies call this method to adapt the architecture
        *before* processing each experience. Strategies also update the
        optimizer automatically.

        .. warning::
            As a general rule, you should NOT use this method to train the
            model. The dataset should be used only to check conditions which
            require the model's adaptation, such as the discovery of new
            classes or tasks.

        .. warning::
            This function only adapts the current module, to recursively adapt all
            submodules use self.recursive_adaptation() instead

        :param experience: the current experience.
        :return:
        """
        pass

    @property
    def _adaptation_device(self):
        """
        The device to use when expanding (or otherwise adapting)
        the model. Defaults to the current device of the fist
        parameter listed using :meth:`parameters`.
        """
        return next(self.parameters()).device

class SimpleCNN(DynamicModule):
    def __init__(self, num_classes=10, in_features=64, initial_out_features=2, auto_adapt=True):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Linear(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self._auto_adapt = auto_adapt
        self.register_buffer("active_units", au_init)

    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param experience: data from the current experience.
        :return:
        """
        super().adaptation(experience)
        device = self._adaptation_device
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        curr_classes = experience.classes_in_this_experience
        print("Current classes: ", len(curr_classes))
        new_nclasses = max(self.classifier.out_features, len(curr_classes)+self.classifier.out_features)
        print("New classes: ", new_nclasses)

        # update classifier weights
        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses).to(device)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b
        
    @property
    def _adaptation_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



