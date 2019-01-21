import torch.nn as nn
import torch


class ElasticHighway(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(ElasticHighway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(output_size if i else input_size, output_size)
                                        for i in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(output_size if i else input_size, output_size)
                                     for i in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(output_size if i else input_size, output_size)
                                   for i in range(num_layers)])
        self.f = nn.ReLU()

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x
