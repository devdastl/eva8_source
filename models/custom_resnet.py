import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
  def __init__(self, in_channel, out_channel, stride=1,dropout=0.05):
    super(ResBlock, self).__init__()
    
    self.sub_layer_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1, bias=False),
                                     nn.BatchNorm2d(in_channel),
                                     nn.ReLU(),
                                    nn.Dropout(dropout)
                                    )
    
    self.sub_layer_2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU(),
                                     nn.Dropout(dropout)
                                    )
                                    
    
  def forward(self, x):
    x = self.sub_layer_1(x)
    x = self.sub_layer_2(x)
    return x


class CustomResNet(nn.Module):
  def __init__(self, dropout=0.05):
    super(CustomResNet, self).__init__()

    self.prep_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False, stride=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(), # 32x32x3 | 32x32x64 | 3x3
                                    )
    
    self.layer1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                               nn.Dropout(dropout)
                               )
    self.R1 = ResBlock(in_channel=128, out_channel=128)

    self.layer2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Dropout(dropout)
                                )
    
    self.layer3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                               nn.Dropout(dropout)
                               )
    self.R2 = ResBlock(in_channel=512, out_channel=512, dropout=0.0)

    self.maxpool_4 = nn.MaxPool2d(kernel_size=4, stride=4)
    self.linear_layer = nn.Linear(512, 10, bias=False)


  def forward(self, x):
    #prep layer
    x = self.prep_layer(x)

    x = self.layer1(x)
    res1 = self.R1(x)
    x = res1 + x

    x = self.layer2(x)

    x = self.layer3(x)
    res2 = self.R2(x)
    x = res2 + x

    x = self.maxpool_4(x)
    x = x.view(x.size(0), -1)
    x = self.linear_layer(x)
    
    #We don't use softmax here because CrossEntropyLoss() function will automatically apply softmax. So returning softmax() of logit
    # will be equivalent of applying softmax twice.
    #return F.softmax(x, dim=-1)
    return x