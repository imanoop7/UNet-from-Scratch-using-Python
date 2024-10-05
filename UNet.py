import numpy as np
import matplotlib.pyplot as plt

# Conv2D: 2D Convolutional layer
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        # Initialize the convolutional layer parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Initialize weights with small random values
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        # Initialize biases with zeros
        self.bias = np.zeros((out_channels, 1))

    def forward(self, input):
        # Calculate output dimensions
        h_out = (input.shape[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_out = (input.shape[3] - self.kernel_size + 2 * self.padding) // self.stride + 1
        output = np.zeros((input.shape[0], self.weights.shape[0], h_out, w_out))

        # Add padding to the input
        padded_input = np.pad(input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Perform convolution operation
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                input_slice = padded_input[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.weights.shape[0]):
                    output[:, k, i, j] = np.sum(input_slice * self.weights[k], axis=(1,2,3)) + self.bias[k]

        return output

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# MaxPool2D: 2D Max Pooling layer
class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        n, c, h, w = input.shape
        h_out = (h - self.kernel_size) // self.stride + 1
        w_out = (w - self.kernel_size) // self.stride + 1
        output = np.zeros((n, c, h_out, w_out))

        # Perform max pooling operation
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                input_slice = input[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(input_slice, axis=(2, 3))

        return output

# Upsample: Upsampling layer
class Upsample:
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def forward(self, input):
        return np.repeat(np.repeat(input, self.scale_factor, axis=2), self.scale_factor, axis=3)

# UNet: Main U-Net architecture
class UNet:
    def __init__(self, n_channels, n_classes):
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (Contracting Path)
        # Each step in the encoder consists of two convolutions followed by a max pooling
        self.conv1 = Conv2D(n_channels, 64)
        self.conv2 = Conv2D(64, 64)
        self.pool1 = MaxPool2D()

        self.conv3 = Conv2D(64, 128)
        self.conv4 = Conv2D(128, 128)
        self.pool2 = MaxPool2D()

        self.conv5 = Conv2D(128, 256)
        self.conv6 = Conv2D(256, 256)
        self.pool3 = MaxPool2D()

        self.conv7 = Conv2D(256, 512)
        self.conv8 = Conv2D(512, 512)
        self.pool4 = MaxPool2D()

        # Bridge
        self.conv9 = Conv2D(512, 1024)
        self.conv10 = Conv2D(1024, 1024)

        # Decoder (Expanding Path)
        # Each step in the decoder consists of an upsampling operation followed by two convolutions
        self.up1 = Upsample()
        self.conv11 = Conv2D(1024, 512)  # Reduce channels before concatenation
        self.conv12 = Conv2D(1024, 512)  # 1024 channels after concatenation

        self.up2 = Upsample()
        self.conv13 = Conv2D(512, 256)  # Reduce channels before concatenation
        self.conv14 = Conv2D(512, 256)  # 512 channels after concatenation

        self.up3 = Upsample()
        self.conv15 = Conv2D(256, 128)  # Reduce channels before concatenation
        self.conv16 = Conv2D(256, 128)  # 256 channels after concatenation

        self.up4 = Upsample()
        self.conv17 = Conv2D(128, 64)  # Reduce channels before concatenation
        self.conv18 = Conv2D(128, 64)  # 128 channels after concatenation

        # Final convolution to produce the output
        self.conv19 = Conv2D(64, n_classes, kernel_size=1)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        
        # Encoder
        conv1 = relu(self.conv1.forward(x))
        conv2 = relu(self.conv2.forward(conv1))
        pool1 = self.pool1.forward(conv2)
        print(f"After first pooling: {pool1.shape}")

        conv3 = relu(self.conv3.forward(pool1))
        conv4 = relu(self.conv4.forward(conv3))
        pool2 = self.pool2.forward(conv4)
        print(f"After second pooling: {pool2.shape}")

        conv5 = relu(self.conv5.forward(pool2))
        conv6 = relu(self.conv6.forward(conv5))
        pool3 = self.pool3.forward(conv6)
        print(f"After third pooling: {pool3.shape}")

        conv7 = relu(self.conv7.forward(pool3))
        conv8 = relu(self.conv8.forward(conv7))
        pool4 = self.pool4.forward(conv8)
        print(f"After fourth pooling: {pool4.shape}")

        # Bridge
        conv9 = relu(self.conv9.forward(pool4))
        conv10 = relu(self.conv10.forward(conv9))
        print(f"After bridge: {conv10.shape}")

        # Decoder
        up1 = self.up1.forward(conv10)
        up1 = self.conv11.forward(up1)  # Reduce channels
        crop_height = min(up1.shape[2], conv8.shape[2])
        crop_width = min(up1.shape[3], conv8.shape[3])
        up1_cropped = up1[:, :, :crop_height, :crop_width]
        conv8_cropped = conv8[:, :, :crop_height, :crop_width]
        merge1 = np.concatenate((up1_cropped, conv8_cropped), axis=1)
        conv12 = relu(self.conv12.forward(merge1))

        up2 = self.up2.forward(conv12)
        up2 = self.conv13.forward(up2)
        crop_height = min(up2.shape[2], conv6.shape[2])
        crop_width = min(up2.shape[3], conv6.shape[3])
        up2_cropped = up2[:, :, :crop_height, :crop_width]
        conv6_cropped = conv6[:, :, :crop_height, :crop_width]
        merge2 = np.concatenate((up2_cropped, conv6_cropped), axis=1)
        conv14 = relu(self.conv14.forward(merge2))

        up3 = self.up3.forward(conv14)
        up3 = self.conv15.forward(up3)
        crop_height = min(up3.shape[2], conv4.shape[2])
        crop_width = min(up3.shape[3], conv4.shape[3])
        up3_cropped = up3[:, :, :crop_height, :crop_width]
        conv4_cropped = conv4[:, :, :crop_height, :crop_width]
        merge3 = np.concatenate((up3_cropped, conv4_cropped), axis=1)
        conv16 = relu(self.conv16.forward(merge3))

        up4 = self.up4.forward(conv16)
        up4 = self.conv17.forward(up4)
        crop_height = min(up4.shape[2], conv2.shape[2])
        crop_width = min(up4.shape[3], conv2.shape[3])
        up4_cropped = up4[:, :, :crop_height, :crop_width]
        conv2_cropped = conv2[:, :, :crop_height, :crop_width]
        merge4 = np.concatenate((up4_cropped, conv2_cropped), axis=1)
        conv18 = relu(self.conv18.forward(merge4))

        conv19 = self.conv19.forward(conv18)
        print(f"Final output shape: {conv19.shape}")

        return conv19

# Example usage
unet = UNet(n_channels=1, n_classes=2)
input_image = np.random.randn(1, 1, 572, 572)
output = unet.forward(input_image)
# Print input and output shapes
print(f"Input shape: {input_image.shape}")
print(f"Output shape: {output.shape}")

# Visualize input and output
plt.figure(figsize=(12, 6))

# Plot the input image
plt.subplot(1, 2, 1)
plt.imshow(input_image[0, 0], cmap='gray')
plt.title('Input Image')
plt.axis('off')

# Plot the output (first channel)
plt.subplot(1, 2, 2)
plt.imshow(output[0, 0], cmap='viridis')
plt.title('Output (Class 1)')
plt.axis('off')

plt.tight_layout()
plt.show()