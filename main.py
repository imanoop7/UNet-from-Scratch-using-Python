# Import necessary libraries
import numpy as np
from UNet import UNet
import matplotlib.pyplot as plt

# Create a simple test image (1 channel, 572x572)
# The image is black (0) with a white (1) square in the middle
test_image = np.zeros((1, 1, 572, 572))
test_image[0, 0, 200:400, 200:400] = 1  # Create a white square in the middle

# Initialize the UNet model with 1 input channel and 2 output classes
unet = UNet(n_channels=1, n_classes=2)

# Perform a forward pass through the network
output = unet.forward(test_image)

# Print input and output shapes
print(f"Input shape: {test_image.shape}")
print(f"Output shape: {output.shape}")

# Visualize input and output
plt.figure(figsize=(12, 6))

# Plot the input image
plt.subplot(1, 2, 1)
plt.imshow(test_image[0, 0], cmap='gray')
plt.title('Input Image')
plt.axis('off')

# Plot the output (first channel)
plt.subplot(1, 2, 2)
plt.imshow(output[0, 0], cmap='viridis')
plt.title('Output (Class 1)')
plt.axis('off')

plt.tight_layout()
plt.show()
