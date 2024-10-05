import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros((out_channels, 1))

    def forward(self, input):
        h_out = (input.shape[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_out = (input.shape[3] - self.kernel_size + 2 * self.padding) // self.stride + 1
        output = np.zeros((input.shape[0], self.weights.shape[0], h_out, w_out))

        padded_input = np.pad(input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

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

def relu(x):
    return np.maximum(0, x)

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        n, c, h, w = input.shape
        h_out = (h - self.kernel_size) // self.stride + 1
        w_out = (w - self.kernel_size) // self.stride + 1
        output = np.zeros((n, c, h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                input_slice = input[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(input_slice, axis=(2, 3))

        return output

class Upsample:
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def forward(self, input):
        return input.repeat(self.scale_factor, axis=2).repeat(self.scale_factor, axis=3)

class UNet:
    def __init__(self, n_channels, n_classes):
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (Contracting Path)
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
        self.up1 = Upsample()
        self.conv11 = Conv2D(1024 + 512, 512)
        self.conv12 = Conv2D(512, 512)

        self.up2 = Upsample()
        self.conv13 = Conv2D(512 + 256, 256)
        self.conv14 = Conv2D(256, 256)

        self.up3 = Upsample()
        self.conv15 = Conv2D(256 + 128, 128)
        self.conv16 = Conv2D(128, 128)

        self.up4 = Upsample()
        self.conv17 = Conv2D(128 + 64, 64)
        self.conv18 = Conv2D(64, 64)

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
        merge1 = np.concatenate((up1, conv8), axis=1)
        conv11 = relu(self.conv11.forward(merge1))
        conv12 = relu(self.conv12.forward(conv11))
        print(f"After first upsampling: {conv12.shape}")

        up2 = self.up2.forward(conv12)
        merge2 = np.concatenate((up2, conv6), axis=1)
        conv13 = relu(self.conv13.forward(merge2))
        conv14 = relu(self.conv14.forward(conv13))
        print(f"After second upsampling: {conv14.shape}")

        up3 = self.up3.forward(conv14)
        merge3 = np.concatenate((up3, conv4), axis=1)
        conv15 = relu(self.conv15.forward(merge3))
        conv16 = relu(self.conv16.forward(conv15))
        print(f"After third upsampling: {conv16.shape}")

        up4 = self.up4.forward(conv16)
        merge4 = np.concatenate((up4, conv2), axis=1)
        conv17 = relu(self.conv17.forward(merge4))
        conv18 = relu(self.conv18.forward(conv17))
        print(f"After fourth upsampling: {conv18.shape}")

        conv19 = self.conv19.forward(conv18)
        print(f"Final output shape: {conv19.shape}")

        return conv19

# Example usage
unet = UNet(n_channels=1, n_classes=2)
input_image = np.random.randn(1, 1, 572, 572)
output = unet.forward(input_image)
print(f"Output shape: {output.shape}")

