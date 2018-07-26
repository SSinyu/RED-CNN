import torch.nn as nn

# RED-CNN
class RED_CNN(nn.Module):
    def __init__(self, out_channels=96):
        super(RED_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, out_channels, kernel_size=5, stride=1, padding=0)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.deconv_last = nn.ConvTranspose2d(out_channels, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        residual1 = x.clone()
        layer = self.relu(self.conv_first(x))
        layer = self.relu(self.conv(layer))
        residual2 = layer.clone()
        layer = self.relu(self.conv(layer))
        layer = self.relu(self.conv(layer))
        residual3 = layer.clone()
        layer = self.relu(self.conv(layer))
        # decoder
        layer = self.deconv(layer)
        layer += residual3
        layer = self.deconv(self.relu(layer))
        layer = self.deconv(self.relu(layer))
        layer += residual2
        layer = self.deconv(self.relu(layer))
        layer = self.deconv_last(self.relu(layer))
        layer += residual1
        layer = self.relu(layer)
        return layer
