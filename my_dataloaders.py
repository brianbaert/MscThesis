import os
from PIL import Image
from collections import Counter

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
