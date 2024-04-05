import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import List, Optional
from torch.nn import Module
import avalanche as avl
import avalanche.models
from avalanche.models import NCMCLassifier
from avalanche.models import SCRModel
from avalanche.evaluation import metrics as metrics
from avalanche.benchmarks.scenarios import CLExperience
from avalanche.benchmarks.utils.flat_data import ConstantSequence
from torch.nn.functional import relu, avg_pool2d
from avalanche.models import BaseModel, MultiHeadClassifier, MultiTaskModule, DynamicModule
from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)

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

# SimpleCNN_32y32 architecture
# -------------------------------
# This class defines a simple convolutional neural network (CNN) for image classification.
# It is designed to work with RGB images of size 224x224 pixels.

class SimpleCNN_32by32(DynamicModule):
    def __init__(self, num_classes=22, in_features=64, initial_out_features=2, auto_adapt=True):
        super(SimpleCNN_32by32, self).__init__()
        
        # Feature extraction layers
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
        # Linear classifier
        self.classifier = nn.Linear(in_features, initial_out_features)
        # Internal tracking of active units
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self._auto_adapt = auto_adapt
        self.register_buffer("active_units", au_init)

    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        """
        If `dataset` contains unseen classes, the classifier is expanded.
        Args:
            experience (CLExperience): Data from the current experience.
        Returns:
            None
        """
        super().adaptation(experience)
        device = self._adaptation_device
        # Retrieve the existing classifier's input features and output features
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        # Determines the new number of classes based on the current experience
        curr_classes = experience.classes_in_this_experience
        #print("Current classes: ", len(curr_classes))
        new_nclasses = max(self.classifier.out_features, len(curr_classes)+self.classifier.out_features)
        #print("New classes: ", new_nclasses)

        # update classifier weights
        # if the number of classes remains the same, no adaptation is needed.
        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        # Creation of a new classifier with updated output features
        # copies the weights and biases from the old classifier to the new one
        # the adapted classifier is placed on the same device
        self.classifier = torch.nn.Linear(in_features, new_nclasses).to(device)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b
        
    @property
    def _adaptation_device(self):
        """
        Returns the device (CPU or GPU) of the first parameter in the model.

        Returns:
            torch.device: The device where the model's parameters reside.
        """   
        return next(self.parameters()).device

    def forward(self, x):
        """
        Forward pass of the neural network model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output predictions.
        """    
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# SimpleCNN_224by224 architecture
# -------------------------------
# This class defines a simple convolutional neural network (CNN) for image classification.
# It is designed to work with RGB images of size 224x224 pixels.

class SimpleCNN_224by224(DynamicModule):
    def __init__(self, num_classes=22, in_features=64, initial_out_features=2, auto_adapt=True):
        super(SimpleCNN_224by224, self).__init__()
        # Feature extraction layers
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
            nn.AdaptiveMaxPool2d(1), # Produces a fixed-size output
            nn.Dropout(p=0.25),
        )
        # Linear classifier
        self.classifier = nn.Linear(in_features, initial_out_features)
        # Internal tracking of active units
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self._auto_adapt = auto_adapt
        self.register_buffer("active_units", au_init)

    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        """
        If `dataset` contains unseen classes, the classifier is expanded.
        Args:
            experience (CLExperience): Data from the current experience.
        Returns:
            None
        """
        super().adaptation(experience)
        device = self._adaptation_device
        # Retrieve the existing classifier's input features and output features
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        # Determines the new number of classes based on the current experience
        curr_classes = experience.classes_in_this_experience
        #print("Current classes: ", len(curr_classes))
        new_nclasses = max(self.classifier.out_features, len(curr_classes)+self.classifier.out_features)
        #print("New classes: ", new_nclasses)

        # update classifier weights
        # if the number of classes remains the same, no adaptation is needed.
        if old_nclasses == new_nclasses:
            return
        # Creation of a new classifier with updated output features
        # copies the weights and biases from the old classifier to the new one
        # the adapted classifier is placed on the same device
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses).to(device)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b
        
    @property
    def _adaptation_device(self):
        """
        Returns the device (CPU or GPU) of the first parameter in the model.

        Returns:
            torch.device: The device where the model's parameters reside.
        """      
        return next(self.parameters()).device

    def forward(self, x):
        """
        Forward pass of the neural network model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output predictions.
        """        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MultiHeadMLP(MultiTaskModule):
    def __init__(self, input_size=32*32*3, hidden_size=256, hidden_layers=2,
                 drop_rate=0, relu_act=True):
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)
        self.classifier = MultiHeadClassifier(hidden_size)

    def forward(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


class MLP(nn.Module, BaseModel):
    def __init__(self, input_size=32*32*3, hidden_size=256, hidden_layers=2,
                 output_size=10, drop_rate=0, relu_act=True, initial_out_features=0):
        """
        :param initial_out_features: if >0 override output size and build an
            IncrementalClassifier with `initial_out_features` units as first.
        """
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)

        if initial_out_features > 0:
            self.classifier = avalanche.models.IncrementalClassifier(in_features=hidden_size,
                                                                     initial_out_features=initial_out_features)
        else:
            self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        return self.features(x)


class SI_CNN(MultiTaskModule):
    def __init__(self, hidden_size=512):
        super().__init__()
        layers = nn.Sequential(*(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Flatten(),
                                 nn.Linear(2304, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5)
                                 ))
        self.features = nn.Sequential(*layers)
        self.classifier = MultiHeadClassifier(hidden_size, initial_out_features=10)

    def forward(self, x, task_labels):
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


class FlattenP(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''

    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class MLP_gss(nn.Module):
    def __init__(self, sizes, bias=True):
        super(MLP_gss, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            if i < (len(sizes)-2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))

        self.net = nn.Sequential(FlattenP(), *layers)

    def forward(self, x):
        return self.net(x)

