import os
import torch
from pathlib import Path
import PIL
from PIL import Image
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder, ImageFolder
import numpy as np
from torchvision import transforms
import my_transformations
import my_utils

class GravitySpy_dataset(ImageFolder):
  def __init__(self, root, cls, transform=None):
    self.data_dir = root
    self.classes = cls
    self.class_to_indx = {c: i for i, c in enumerate(self.classes)}
    self.image_paths=[]
    self.labels=[]
    for class_name in self.classes:
      class_path = os.path.join(root, class_name)
      for filename in os.listdir(class_path):
          self.image_paths.append(os.path.join(class_path, filename))
          self.labels.append(self.class_to_indx[class_name])
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path)
    image = image.convert('RGB')
    label = self.labels[idx]
    #target = self.targets[idx]

    if self.transform:
      image = self.transform(image)

    #return image, label
    return image, label

  def count_class_instances(self):
    label_counts = Counter(self.labels)
    return label_counts

class GravitySpy_1_0_dataset(ImageFolder):
  def __init__(self, root, cls, transform=None):
    self.data_dir = root
    self.classes = cls
    self.class_to_indx = {c: i for i, c in enumerate(self.classes)}
    self.image_paths=[]
    self.labels=[]
    for class_name in self.classes:
      class_path = os.path.join(root, class_name)
      for filename in os.listdir(class_path):
        if filename.endswith("1.0.png"):
          self.image_paths.append(os.path.join(class_path, filename))
          self.labels.append(self.class_to_indx[class_name])
          #self.targets.append(self.class_to_indx[class_name])
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path)
    image = image.convert('RGB')
    label = self.labels[idx]
    #target = self.targets[idx]

    if self.transform:
      image = self.transform(image)

    #return image, label
    return image, label

  def count_class_instances(self):
    label_counts = Counter(self.labels)
    return label_counts

class GravitySpy_0_5_dataset(ImageFolder):
  def __init__(self, root, cls, transform=None):
    self.data_dir = root
    self.classes = cls
    self.class_to_indx = {c: i for i, c in enumerate(self.classes)}
    self.image_paths=[]
    self.labels=[]
    for class_name in self.classes:
      class_path = os.path.join(root, class_name)
      for filename in os.listdir(class_path):
        if filename.endswith("0.5.png"):
          self.image_paths.append(os.path.join(class_path, filename))
          self.labels.append(self.class_to_indx[class_name])
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path)
    image = image.convert('RGB')
    label = self.labels[idx]

    if self.transform:
      image = self.transform(image)

    return image, label

  def count_class_instances(self):
    label_counts = Counter(self.labels)
    return label_counts

class GravitySpy_2_0_dataset(ImageFolder):
  def __init__(self, root, cls, transform=None):
    self.data_dir = root
    self.classes = cls
    self.class_to_indx = {c: i for i, c in enumerate(self.classes)}
    self.image_paths=[]
    self.labels=[]
    for class_name in self.classes:
      class_path = os.path.join(root, class_name)
      for filename in os.listdir(class_path):
        if filename.endswith("2.0.png"):
          self.image_paths.append(os.path.join(class_path, filename))
          self.labels.append(self.class_to_indx[class_name])
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path)
    image = image.convert('RGB')
    label = self.labels[idx]

    if self.transform:
      image = self.transform(image)

    return image, label

  def count_class_instances(self):
    label_counts = Counter(self.labels)
    return label_counts

class GravitySpy_4_0_dataset(ImageFolder):
  def __init__(self, root, cls, transform=None):
    self.data_dir = root
    self.classes = cls
    self.class_to_indx = {c: i for i, c in enumerate(self.classes)}
    self.image_paths=[]
    self.labels=[]
    for class_name in self.classes:
      class_path = os.path.join(root, class_name)
      for filename in os.listdir(class_path):
        if filename.endswith("4.0.png"):
          self.image_paths.append(os.path.join(class_path, filename))
          self.labels.append(self.class_to_indx[class_name])
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path)
    image = image.convert('RGB')
    label = self.labels[idx]

    if self.transform:
      image = self.transform(image)

    return image, label

  def count_class_instances(self):
    label_counts = Counter(self.labels)
    return label_counts


class MultiViewGravitySpyDataset(ImageFolder):
  def __init__(self, root, cls, transform=None):
    self.data_dir = root
    self.classes = cls
    self.class_to_indx = {c: i for i, c in enumerate(self.classes)}
    self.image_paths = []
    self.labels = []
    self.versions = ["0.5.png", "1.0.png", "2.0.png", "4.0.png"]

    for class_name in self.classes:
      class_path = os.path.join(root, class_name)
      for filename in os.listdir(class_path):
        if any(filename.endswith(version) for version in self.versions):
          self.image_paths.append(os.path.join(class_path, filename))
          self.labels.append(self.class_to_indx[class_name])

    self.transform = transform

  def __len__(self):
    return len(self.image_paths) // 4  # Assuming 4 versions for each image

  def __getitem__(self, idx):
    image_paths = self.image_paths[idx*4: (idx+1)*4]  # Get paths for the 4 versions
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
    # Apply transformations if provided
    if self.transform:
      images = [self.transform(img) for img in images]

    # Concatenate images
    top_row = Image.fromarray(np.concatenate([np.array(images[0]), np.array(images[1])], axis=1))
    bottom_row = Image.fromarray(np.concatenate([np.array(images[2]), np.array(images[3])], axis=1))
    final_image = Image.fromarray(np.concatenate([np.array(top_row), np.array(bottom_row)], axis=0))
    temp = np.array(final_image)

    temp = temp.astype(np.float32)
    temp /= 255.0
    fused_image = torch.from_numpy(temp.transpose((2, 0, 1)))

    # Convert back to PIL Image if needed for further processing
    # fused_image = Image.fromarray(fused_image)  # Uncomment if required
    label = self.labels[idx*4]  # Assuming labels are the same for all versions
    return fused_image, label

class FractalDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        return data, label, idx

class FractalImages(ImageFolder):
  def __init__(self, root, cls, transform=None):
    self.data_dir = root
    self.classes = cls
    self.class_to_indx = {c: i for i, c in enumerate(self.classes)}
    self.image_paths = []
    self.labels = []
    for class_name in self.classes:
      class_path = os.path.join(root, class_name)
      for filename in os.listdir(class_path):
          image_path = os.path.join(class_path, filename)
          self.image_paths.append(image_path)
          self.labels.append(self.class_to_indx[class_name])

    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path)
    image = image.convert('RGB')
    label = self.labels[idx]
    if self.transform:
      image = self.transform(image)

    temp = np.array(image)
    temp = temp.astype(np.float32)
    temp /= 255.0
    image = torch.from_numpy(temp.transpose((2, 0, 1)))

    #extract filename needed for other dataloader
    filename = os.path.splitext(os.path.basename(image_path))[0]
    return image, label, filename

  def count_class_instances(self):
    label_counts = Counter(self.labels)
    return label_counts

class MultimodalFractalImages(ImageFolder):
  def __init__(self, root, cls, fd, transform=None):
    self.data_dir = root
    self.classes = cls
    self.class_to_indx = {c: i for i, c in enumerate(self.classes)}
    self.fd_matrix = fd
    self.image_paths = []
    self.labels = []
    for class_name in self.classes:
      class_path = os.path.join(root, class_name)
      for filename in os.listdir(class_path):
          image_path = os.path.join(class_path, filename)
          self.image_paths.append(image_path)
          self.labels.append(self.class_to_indx[class_name])

    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path)
    image = image.convert('RGB')
    label = self.labels[idx]
    if self.transform:
      image = self.transform(image)

    temp = np.array(image)
    temp = temp.astype(np.float32)
    temp /= 255.0
    image = torch.from_numpy(temp.transpose((2, 0, 1)))

    #extract filename needed for other dataloader
    filename = os.path.splitext(os.path.basename(image_path))[0]
    index = my_utils.find_index(filename)
    index = index - 896
    image_fd = self.fd_matrix[index]
    return image, label, image_fd

  def count_class_instances(self):
    label_counts = Counter(self.labels)
    return label_counts

