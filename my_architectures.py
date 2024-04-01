import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import List, Optional
from torch.nn import Module
from avalanche.benchmarks.scenarios import CLExperience
from avalanche.benchmarks.utils.flat_data import ConstantSequence
from torch.nn.functional import relu, avg_pool2d
from avalanche.models import MultiHeadClassifier, MultiTaskModule, DynamicModule
from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)

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
    def __init__(self, num_classes=10, in_features=64, initial_out_features=2, auto_adapt=True):
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
    def __init__(self, num_classes=10, in_features=64, initial_out_features=2, auto_adapt=True):
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

    
"""This is the slimmed ResNet as used by Lopez et al. in the GEM paper."""
# THIS IS NOT WORKING YET, NEEDS TO BE ADJUSTED SO THE ADAPTATION METHOD IS PROVIDED

class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(DynamicModule):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SlimResNet18(nclasses, nf=20):
    """Slimmed ResNet18."""
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)


class MTSlimResNet18(MultiTaskModule, DynamicModule):
    """MultiTask Slimmed ResNet18."""

def __init__(self, nclasses, nf=20):
    super().__init__()
    self.in_planes = nf
    block = BasicBlock
    num_blocks = [2, 2, 2, 2]

    self.conv1 = conv3x3(3, nf * 1)
    self.bn1 = nn.BatchNorm2d(nf * 1)
    self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
    self.linear = MultiHeadClassifier(nf * 8 * BasicBlock.expansion, nclasses)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, task_labels):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out, task_labels)
        return out

