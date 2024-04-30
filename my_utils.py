import torch
import umap
import numba
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def checkpoint(model, filename):
    # Save the current state of the model to a file
    torch.save(model.state_dict(), filename)
    print("Saved Pytorch model state to ", filename)

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

def classes_to_indices(classes):
    return {c: i for i, c in enumerate(classes)}

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

def extract_glitchName_from_filenames(filenames, output_filename):
  with open(output_filename, 'w') as output_file:
    for filename in filenames:
      parts = filename.split('_')
      if len(parts) >= 4 and parts[1] == 'calculation':
        gN = parts[2]
        output_file.write(gN + '\n')

@timeit    
def n_test_predictions(model, data_loader, classes, n):
    temp = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    for i, sample in data_loader:
        i, sample = i.to(device), sample.to(device)
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

def plot_first_image(dataloader):
    """
    Plots the first image from a given dataloader.
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing image data.
    Returns:
        None
    """
    # Get the first batch from the dataloader
    batch = next(iter(dataloader))

    # Extract the first image and its label (if available)
    image, label = batch

    # Convert image tensor to numpy array
    image_np = image[0].numpy()

    # Create a subplot and display the image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_np.transpose(1, 2, 0))  # Transpose to (H, W, C) format
    plt.axis('off')  # Hide axes
    plt.title(f"Label: {label[0]}") if label is not None else plt.title("First Image")
    plt.show()
    # Example usage:
    # Assuming you have a DataLoader called 'my_dataloader'
    # plot_first_image(my_dataloader)

def plot_batch_of_images(dataloader, nrow=4, padding=2):
    for batch in dataloader:
        images, labels = batch
        grid = torchvision.utils.make_grid(images, nrow, padding, normalize=True)
        grid_np = grid.numpy().transpose((1,2,0))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.show()
        break

def plot_confusion_matrix(cm, classes, name):
    plt.figure(figsize=(10,10))
    # Use seaborn heatmap for visualization
    sns.heatmap(cm, annot=True, cmap='viridis', fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    figTemp = plt.gcf()
    plt.show()
    plt.draw()
    figTemp.savefig(name)
    plt.close()

def plot_f1_scores(f1, classes, name):
    # Create a horizontal bar plot for F1 scores with different colors
    plt.figure(figsize=(10,7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    plt.barh(np.arange(len(classes)), f1, color=colors, align='center', alpha=0.5)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel('F1 Score')
    plt.title('F1 Score for Each Class')
    figTemp = plt.gcf()
    plt.show()
    plt.draw()
    figTemp.savefig(name)
    plt.close()

def plot_f1_scores_with_labels(f1, classes, name):
  # Create a horizontal bar plot for F1 scores with different colors
  plt.figure(figsize=(10, 7))
  colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

  # Create rectangles for bars
  bars = plt.barh(np.arange(len(classes)), f1, color=colors, align='center')

  # Add F1 values as text labels on top of bars
  for bar, value in zip(bars, f1):
    plt.text(value + 0.05, bar.get_height() / 2, f"{value:.2f}", va='center')

  plt.yticks(np.arange(len(classes)), classes)
  plt.xlabel('F1 Score')
  plt.title('F1 Score for Each Class')
  figTemp = plt.gcf()
  plt.show()
  plt.draw()
  figTemp.savefig(name)
  plt.close()


def cl_adaptive_train_loop(bm, cl_strategy, model, optimizer, number_of_workers, classes, scr=False):
    results = []
    print('Starting experiment with strategy:', cl_strategy)
    for experience in bm.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print(len(experience.classes_in_this_experience))
        cl_strategy.train(experience)
        print('Training completed')
        print("Shape of the FC layer: ")
        print(model.classifier)

        # Get classification layer weights after training
        if scr==False:
            classification_weights = model.fc3.weight.detach().numpy()
        else:
            classification_weights = model.feature_extractor.fc3.weight.detach().numpy()

        print(classification_weights)
        print(classes)
        # Create a DataFrame for seaborn violin plot
        weight_df = pd.DataFrame(classification_weights.T, columns=classes)
        
        # Set up seaborn style
        sns.set(style="whitegrid")
        
        # Create a violin plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=weight_df, palette="viridis", inner="quartile")

        plt.xlabel("Class")
        plt.ylabel("Weight Value")
        plt.title("Violin Plot of Classification Layer Weight changes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
                
        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(bm.test_stream))


    return results

@timeit
def cl_simple_train_loop(bm, cl_strategy, model, optimizer, number_of_workers, classes, name, scr=False):
    results = []
    # Get initial classification layer weights after training
    if torch.cuda.is_available():
        # Get initial classification layer weights after training
        if scr==False:
            init_weights = np.copy(model.fc3.weight.cpu().detach().numpy())
        else:
            init_weights = np.copy(model.feature_extractor.fc3.weight.cpu().detach().numpy())
    else:
        if scr==False:
            init_weights = np.copy(model.fc3.weight.detach().numpy())
        else:
            init_weights = np.copy(model.feature_extractor.fc3.weight.detach().numpy())
    print('Starting experiment with strategy:', cl_strategy)
    
    for experience in bm.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train one experience
        res = cl_strategy.train(experience)
        print("Training completed")

        if torch.cuda.is_available():
            # Get initial classification layer weights after training
            if scr==False:
                classification_weights = np.copy(model.fc3.weight.cpu().detach().numpy())
            else:
                classification_weights = np.copy(model.feature_extractor.fc3.weight.cpu().detach().numpy())
        else:
            # Get classification layer weights after training
            if scr==False:
                classification_weights = np.copy(model.fc3.weight.detach().numpy())
            else:
                classification_weights = np.copy(model.feature_extractor.fc3.weight.detach().numpy())

        temp = init_weights - classification_weights        
        # Create a DataFrame for seaborn violin plot
        weight_df = pd.DataFrame(temp.T, columns=classes)
        
        # Set up seaborn style
        sns.set(style="whitegrid")
        
        # Create a violin plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=weight_df, palette="viridis", inner="quartile")
        
        plt.xlabel("Class")
        plt.ylabel("Weight Value")
        plt.title(f"Violin Plot - Exp {experience.current_experience} - {name}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        figTemp = plt.gcf()
        plt.show()
        plt.draw()
        filename = f"{name}_exp_{experience.current_experience}.png"
        figTemp.savefig(filename)
        plt.close()

        # Create a DataFrame for seaborn violin plot
        weight_df = pd.DataFrame(classification_weights.T, columns=classes)
        
        """print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(bm.test_stream))"""
        # Get initial classification layer weights after training
        if torch.cuda.is_available():
            # Get initial classification layer weights after training
            if scr==False:
                init_weights = np.copy(model.fc3.weight.cpu().detach().numpy())
            else:
                init_weights = np.copy(model.feature_extractor.fc3.weight.cpu().detach().numpy())
        else:
            if scr==False:
                init_weights = np.copy(model.fc3.weight.detach().numpy())
            else:
                init_weights = np.copy(model.feature_extractor.fc3.weight.detach().numpy())
    all_metrics = cl_strategy.evaluator.get_all_metrics()
    print(f"Stored metrics: {list(all_metrics.keys())}")
    return results

@timeit
def plot_tSNE_data_embedding(model, dataloader, classes, name):
    # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize lists to store embeddings and labels
    embeddings = []
    labels = []

    # Iterate over the data and collect embeddings
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        if torch.cuda.is_available():
            embeddings.append(outputs.cpu().detach().numpy())
            labels.append(targets.cpu().detach().numpy())
        else:
            embeddings.append(outputs.detach().numpy())
            labels.append(targets.detach().numpy())


    # Concatenate embeddings and labels
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    # Create a scatter plot of the t-SNE visualization
    plt.figure(figsize=(10, 8))

    # Define distinct colors for each class
    class_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'gray']

    # Scatter plot with different colors for each class
    for class_label in np.unique(labels):
        plt.scatter(embeddings_tsne[labels == class_label, 0],
                    embeddings_tsne[labels == class_label, 1],
                    label=classes[int(class_label)],
                    c=class_colors[int(class_label)],  # Assign distinct color
                    edgecolor='k')

    plt.title('t-SNE Visualization of Data Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()  # Show legend with all class labels

    figTemp = plt.gcf()
    plt.show()
    plt.draw()
    figTemp.savefig(name)
    plt.close()

@timeit
def plot_umap_data_embedding(model, dataloader, classes, reducer, name):
    # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Initialize lists to store embeddings and labels
    embeddings = []
    labels = []

    # Iterate over the data and collect embeddings
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        if torch.cuda.is_available():
            embeddings.append(outputs.cpu().detach().numpy())
            labels.append(targets.cpu().detach().numpy())
        else:
            embeddings.append(outputs.detach().numpy())
            labels.append(targets.detach().numpy())

    # Concatenate embeddings and labels
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    # Apply umap to reduce dimensionality
    proj = reducer.fit_transform(embeddings)

    # Create a scatter plot of the t-SNE visualization
    plt.figure(figsize=(10, 8))

    # Define distinct colors for each class
    class_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'gray']

    # Scatter plot with different colors for each class
    for class_label in np.unique(labels):
        plt.scatter(proj[labels == class_label, 0],
                    proj[labels == class_label, 1],
                    label=classes[int(class_label)],
                    c=class_colors[int(class_label)],  # Assign distinct color
                    edgecolor='k')

    plt.title('umap Visualization of Data Embeddings')
    plt.xlabel('umap Dimension 1')
    plt.ylabel('umap Dimension 2')
    plt.legend()  # Show legend with all class labels
    
    figTemp = plt.gcf()
    plt.show()
    plt.draw()
    figTemp.savefig(name)
    plt.close()    

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
