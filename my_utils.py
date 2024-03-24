import torch
import numpy as np
import os

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

def get_predictions(model, dataloader):
    # Set the model to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []
    # Disable gradient calculation
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            # Get the index of the maximum value
            _, preds = torch.max(outputs, 1)
            # Append the predictions and labels to the respective lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Return the lists of predictions and labels
    return all_preds, all_labels

def get_classes_from_dir(goal_dir):
    # Get a list of all directory names in the specified directory
    temp_classes = [d for d in next(os.walk(train_dir))[1] if os.path.isdir(os.path.join(train_dir, d))]
    # Return the list of directories, which represent the classes
    return temp_classes
