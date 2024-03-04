import os
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


UPSCALE_FACTOR = 4
CROP_SIZE = 88
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# Now we will start implementing the model.
class ResidualBlock(nn.Module):  #Defines a  class name 'ResidualBlock' that inherits from 'nn.module'
  def __init__(self, channels):   #Constructor of 'ResidualBlock' class
    super(ResidualBlock, self).__init__()   #Calls the constructor of the superclass
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)   #Normalizes the activation
    self.prelu = nn.PReLU()
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)
  def forward(self, x):   #Defines forward-pass of residual block
    residual = self.conv1(x)   #Produces 'intemediate feature map'
    residual = self.bn1 (residual)   #Produces 'normalized feature map'
    residual = self.prelu(residual)   #Produces 'PReLU activated feature map'
    residual = self.conv2(residual)   #Produces 'final residual feature map'
    residual = self.bn2(residual)     #Produces 'residual feature map'
    return x + residual   #O/p of residual block

# We just implemented a pretty standard residual block here

class UpsampleBlock(nn.Module):
  def __init__(self, in_channels, up_scale):
    super(UpsampleBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2,
                          kernel_size=3, padding=1)
    self.pixel_shuffle = nn.PixelShuffle(up_scale)
    self.prelu = nn.PReLU()
  def forward(self, x):
    x = self.conv(x)
    x = self.pixel_shuffle(x)
    x = self.prelu(x)
    return x

class Generator(nn.Module):
  def __init__(self, scale_factor):
    super(Generator, self).__init__()
    upsample_block_num = int(math.log(scale_factor, 2))

    self.block1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=9, padding=4),
        nn.PReLU()
    )

    self.block2 = ResidualBlock(64)
    self.block3 = ResidualBlock(64)
    self.block4 = ResidualBlock(64)
    self.block5 = ResidualBlock(64)
    self.block6 = ResidualBlock(64)
    self.block7 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64)
    )
    block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
    block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
    self.block8 = nn.Sequential(*block8)
  def forward(self, x):
    block1 = self.block1(x)
    block2 = self.block2(block1)
    block3 = self.block3(block2)
    block4 = self.block4(block3)
    block5 = self.block5(block4)
    block6 = self.block6(block5)
    block7 = self.block7(block6)
    block8 = self.block8(block1 + block7)
    return (torch.tanh(block8) + 1) / 2

device  = torch.device('cpu')
# Standard device selectoin
device


# Load the trained model
netG = Generator(UPSCALE_FACTOR)
weights_path = 'D:/minor-project-main/backend/generator_weights77.pth'
netG.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
netG = netG.to(device)  # Move the model to the GPU if available
netG.eval()  # Set the model to evaluation mode

# Now you can proceed with testing
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import os

# Define Test Dataset Class
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
        self.upscale_factor = upscale_factor
        self.transform = ToTensor()

    def __getitem__(self, index):
        # Load image
        img = Image.open(self.image_filenames[index])
        # Apply any preprocessing transforms
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_filenames)

# Set directory path for test data
test_data_dir = "D:/minor-project-main/backend/uploads"

# Set upscale factor
upscale_factor = 4  # Adjust as needed

# Instantiate Test Dataset
test_set = TestDatasetFromFolder(test_data_dir, upscale_factor=upscale_factor)

# Set batch size and number of workers for DataLoader
batch_size = 1  # Adjust as needed
num_workers = 0  # Adjust as needed

# Create DataLoader for Testing
testloader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

# Iterate Over Test DataLoader
for data in testloader:
    # Perform evaluation using the loaded test data
    # Pass each batch of test images through your trained model for evaluation
    # Replace the comment below with your evaluation code
    # For example, you can print the shape of each batch to ensure it is loaded correctly
    print("Batch shape:", data.shape)
    
    
for data_idx, data in enumerate(testloader):
    # Perform evaluation using the loaded test data
    # Pass each batch of test images through your trained model for evaluation
    # Replace the comment below with your evaluation code
    # For example:
    # predicted_hr_images = netG(data.to(device))

    # Convert generated HR images to numpy arrays
    generated_hr_images = netG(data.to(device)).detach().cpu().numpy()

    # Iterate over batch and visualize results
    for i in range(batch_size):
        # Convert tensors to numpy arrays for visualization
        lr_image = data[i].numpy().transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
        hr_image_pred = generated_hr_images[i].transpose((1, 2, 0))  # Convert from CxHxW to HxWxC

        # # Plot original LR image
        # plt.subplot(1, 2, 1)
        # plt.imshow(lr_image)
        # plt.title('Low-Resolution')

        # # Plot predicted HR image
        # plt.subplot(1, 2, 2)
        # plt.imshow(hr_image_pred)
        # plt.title('Generated High-Resolution')

        # plt.show()  # Display the plot

        #Save the generated HR image
        save_path = os.path.join('D:/minor-project-main/backend/processed', "restored.jpg")
        plt.imsave(save_path, hr_image_pred)
        print(hr_image_pred.shape)