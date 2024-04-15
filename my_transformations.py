import torch
import torchvision
import torchvision.transforms as transforms
import my_utils

# Define the bounding box coordinates
top, left, height, width = 60, 100, 480, 580

#Transformations needed
transformGray = transforms.Compose([
     transforms.Resize((140,170)),
     transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     transforms.Normalize((0.5),(0.5))
    ])

transformRGB = transforms.Compose([
     transforms.Resize((140,170)),
     transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

transformAV = transforms.Compose([
     transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5),(0.5))
    ])

transformAV_Crop = transforms.Compose([
     my_utils.CustomCrop(top, left, height, width),
     transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5),(0.5))
    ])

transformAV32 = transforms.Compose([
     transforms.Resize((32,32)),
     transforms.ToTensor()
    ])

transformAV32_Crop = transforms.Compose([
    my_utils.CustomCrop(top, left, height, width),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

transformAV_224_Crop = transforms.Compose([
    my_utils.CustomCrop(top, left, height, width),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

transformAV_224 = transforms.Compose([
     transforms.Resize((224,224)),
     transforms.ToTensor()
     #transforms.Normalize((0.5),(0.5))
    ])

transformAV_Fuse_Crop = transforms.Compose([
    my_utils.CustomCrop(top, left, height, width),
    transforms.Resize((112,112))
])

transformAV_Fuse = transforms.Compose([
    transforms.Resize((112,112))
])

