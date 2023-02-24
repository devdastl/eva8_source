import torch
import torch.nn as nn
import torch.nn.functional as F

class Ultimus(nn.Module):
    def __init__(self, input_neuron = 48, neuron_to_scale = 8, patch_num = 1):
        super(Ultimus, self).__init__()

        self.patch_num = patch_num
        self.neuron_to_scale = neuron_to_scale

        self.K = nn.Linear(input_neuron, neuron_to_scale, bias=False)
        self.Q = nn.Linear(input_neuron, neuron_to_scale, bias=False)
        self.V = nn.Linear(input_neuron, neuron_to_scale, bias=False)

        self.Out = nn.Linear(neuron_to_scale, input_neuron, bias=False)

    def forward(self, x):
        #reshaping first dimension as batch_size and patch_num
        K = self.K(x).view(x.shape[0] // self.patch_num, self.patch_num, self.neuron_to_scale)
        Q = self.Q(x).view(x.shape[0] // self.patch_num, self.patch_num, self.neuron_to_scale)
        V = self.V(x).view(x.shape[0] // self.patch_num, self.patch_num, self.neuron_to_scale)

        Q_tanspose = torch.transpose(Q, 1,2)

        AM = torch.matmul(Q_tanspose, K)/(self.neuron_to_scale**(0.5))
        AM = F.softmax(AM, dim=1)
        Z = torch.matmul(V, AM)

        out = self.Out(Z.squeeze())

        return out


class TransfromerAttention(nn.Module):
    def __init__(self, input_shape=32):
        super(TransfromerAttention, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=1, bias=False),
            nn.Conv2d(32, 48, kernel_size=3, padding=0, stride=1, bias=False),
        )

        self.Final = nn.Linear(48, 10, bias=False)

        self.GAP = nn.AdaptiveAvgPool2d((1,1))

        self.ultimus_b1 = Ultimus()
        self.ultimus_b2 = Ultimus()
        self.ultimus_b3 = Ultimus()
        self.ultimus_b4 = Ultimus()

    def forward(self, x):
        x = self.conv_block(x)
        
        X = self.GAP(x)
        X_conv = torch.squeeze(X)

        #adding x_conv with the output of ultimus block and passing it to the next
        output = self.ultimus_b1(X_conv)
        output = self.ultimus_b2(X_conv + output)
        output = self.ultimus_b3(X_conv + output)
        output = self.ultimus_b4(X_conv + output)

        output = self.Final(output)

        return output
