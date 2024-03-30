import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image

def checkpoint(model, filename):
    # Save the current state of the model to a file
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    # Load the state of the model from a file
    model.load_state_dict(torch.load(filename))

def calculate_accuracy(outputs, labels):
    # Get the index of the maximum value
    _, predicted = torch.max(outputs.data, 1)
    # Calculate the number of correct predictions 
    correct = (predicted == labels).sum().float()
    # Calculate the accuracy by dividing the number of correct predictions by the total number of predictions
    accuracy = correct / len(labels)
    return accuracy

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__}: {(te - ts) * 1000} ms')
        return result
    return timed

@timeit
def get_predictions(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # Set the model to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []
    # Disable gradient calculation
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            # Get the index of the maximum value
            _, preds = torch.max(outputs, 1)
            # Append the predictions and labels to the respective lists
            all_preds.append(preds)
            all_labels.append(labels)
            #all_preds.extend(preds.cpu().numpy())
            #all_labels.extend(labels.cpu().numpy())
    # Return the lists of predictions and labels
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    return all_preds, all_labels

def get_classes_from_dir(goal_dir):
    # Get a list of all directory names in the specified directory
    temp_classes = [d for d in next(os.walk(goal_dir))[1] if os.path.isdir(os.path.join(goal_dir, d))]
    # Return the list of directories, which represent the classes
    return temp_classes
    
def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

@timeit    
def n_test_predictions(model, data_loader, classes, n):
    temp = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    for i, sample in data_loader:
        if temp!=n:
            x = i
            y = sample[0]
            with torch.no_grad():
                pred = model(x)
                predicted, actual = classes[pred[0].argmax(0)], classes[y]
                print(f'Predicted: "{predicted}", Actual: "{actual}"')
                temp+=1
        else:
            break


class CustomCrop(object):
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        """
        Crop the given image.
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return img.crop((self.left, self.top, self.left + self.width, self.top + self.height))

def crop_and_resize_image(img, top, left, height, width, output_size):
    """
    Crop the given image and resize it to the desired output size.
    Args:
        img (PIL Image or Tensor): Image to be cropped.
        top (int): Vertical component of the top-left corner of the crop box.
        left (int): Horizontal component of the top-left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        output_size (sequence or int): Desired output size (same semantics as resize).
    Returns:
        PIL Image or Tensor: Cropped and resized image.
    """
    cropped_img = F.resized_crop(img, top, left, height, width, size=output_size)
    return cropped_img
