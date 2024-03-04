import os
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import math


class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.prelu = nn.PReLU()
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)
  def forward(self, x):
    residual = self.conv1(x)
    residual = self.bn1(residual)
    residual = self.prelu(residual)
    residual = self.conv2(residual)
    residual = self.bn2(residual)
    return x + residual


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

# Define the model architecture (Generator in this case)
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

# Set upscale factor
UPSCALE_FACTOR = 4

# Instantiate the model
model = Generator(UPSCALE_FACTOR)

# Load the trained weights
weights_path = 'C:/ab/remastering/backend/generator_weights77.pth'
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Set input and output folders
input_folder = 'C:/ab/remastering/backend/uploads'
output_folder = 'C:/ab/remastering/backend/processed'

# Iterate over images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        # Load image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Apply any necessary preprocessing
        transform = ToTensor()
        input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Convert the output to a NumPy array and save
        output_array = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_img = Image.fromarray((output_array * 255).astype(np.uint8))

        # Save the high-resolution image to the output folder
        output_path = os.path.join(output_folder, 'restored.png')
        output_img.save(output_path)
        # import matplotlib.pyplot as plt

        # # Display input and output images side by side
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(img))
        # plt.title("Input Image")

        # plt.subplot(1, 2, 2)
        # plt.imshow(output_array)
        # plt.title("Output Image")

        # plt.show()

