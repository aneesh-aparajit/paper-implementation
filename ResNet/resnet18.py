import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('mps' if torch.has_mps else 'cpu')


class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1) -> None:
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layer(x)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool) -> None:
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = ConvLayer(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = ConvLayer(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if self.downsample:
            self.downsample_layer = ConvLayer(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)
        else:
            self.downsample_layer = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample:
            x += residual
            return x
        return x


class ResNet18(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_times: typing.List[int] = [2, 2, 2, 2]) -> None:
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            ConvLayer(in_channels=in_channels,
                      out_channels=64,
                      kernel_size=(7, 7),
                      stride=(2, 2),
                      padding=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        self.conv2_x = self._make_layer(
            in_channels=64, out_channels=64, num_time=num_times[0])
        self.conv3_x = self._make_layer(
            in_channels=64, out_channels=128, num_time=num_times[1])
        self.conv4_x = self._make_layer(
            in_channels=128, out_channels=256, num_time=num_times[2])
        self.conv5_x = self._make_layer(
            in_channels=256, out_channels=512, num_time=num_times[3])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def _make_layer(self,
                    in_channels: int,
                    out_channels: int,
                    num_time: int = 2) -> nn.Sequential:
        layers = []
        if in_channels != out_channels:
            layer = ResidualBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  downsample=True)
            layers.append(layer)
        else:
            layer = ResidualBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  downsample=False)
            layers.append(layer)
        for i in range(num_time-1):
            layer = ResidualBlock(in_channels=out_channels,
                                  out_channels=out_channels,
                                  downsample=False)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f'x before conv1: {x.shape}')
        x = self.conv1(x)
        print(f'x before conv2_x: {x.shape}')
        x = self.conv2_x(x)
        x = self.maxpool(x)
        print(f'x before conv3_x: {x.shape}')
        x = self.conv3_x(x)
        x = self.maxpool(x)
        print(f'x before conv4_x: {x.shape}')
        x = self.conv4_x(x)
        x = self.maxpool(x)
        print(f'x before conv5_x: {x.shape}')
        x = self.conv5_x(x)
        print(f'x before avgpool: {x.shape}')
        x = self.avgpool(x)
        print(f'x before reshaping: {x.shape}')
        x = x.view(-1, 512)
        print(f'x before fc: {x.shape}')
        return self.fc(x)


if __name__ == '__main__':
    model = ResNet18(in_channels=3, num_classes=10, num_times=[2, 2, 2, 2])
    X = torch.rand((32, 3, 224, 224))
    yHat = model.forward(X)
    print(f'''
X: {X.shape}
yHat: {yHat.shape}
Model:
------
{model}''')

'''
Output:
❯ python resnet18.py
x before conv1: torch.Size([32, 3, 224, 224])
x before conv2_x: torch.Size([32, 64, 56, 56])
x before conv3_x: torch.Size([32, 64, 28, 28])
x before conv4_x: torch.Size([32, 128, 14, 14])
x before conv5_x: torch.Size([32, 256, 7, 7])
x before avgpool: torch.Size([32, 512, 7, 7])
x before reshaping: torch.Size([32, 512, 1, 1])
x before fc: torch.Size([32, 512])

X: torch.Size([32, 3, 224, 224])
yHat: torch.Size([32, 10])
Model:
------
ResNet18(
  (conv1): Sequential(
    (0): ConvLayer(
      (conv_layer): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (1): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=1, ceil_mode=False)
  )
  (conv2_x): Sequential(
    (0): ResidualBlock(
      (conv1): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv2): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (downsample_layer): Sequential()
    )
    (1): ResidualBlock(
      (conv1): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv2): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (downsample_layer): Sequential()
    )
  )
  (conv3_x): Sequential(
    (0): ResidualBlock(
      (conv1): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv2): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (downsample_layer): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
    )
    (1): ResidualBlock(
      (conv1): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv2): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (downsample_layer): Sequential()
    )
  )
  (conv4_x): Sequential(
    (0): ResidualBlock(
      (conv1): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv2): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (downsample_layer): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
    )
    (1): ResidualBlock(
      (conv1): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv2): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (downsample_layer): Sequential()
    )
  )
  (conv5_x): Sequential(
    (0): ResidualBlock(
      (conv1): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv2): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (downsample_layer): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
    )
    (1): ResidualBlock(
      (conv1): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv2): ConvLayer(
        (conv_layer): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (downsample_layer): Sequential()
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=1)
  (fc): Linear(in_features=512, out_features=10, bias=True)
  (maxpool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
)
'''
