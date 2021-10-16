import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder

class OursEncoder(torch.nn.Module):
    def __init__(self, input_dims=1, output_dims=320, reduced_dims=160, hidden_dims=64, depth=10, classes=3, sim_fun='euclidean', cate_fun='argmax'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.classes = classes
        self.cate_fun = cate_fun
        self.sim_fun = sim_fun
        self.tau = 0.1
        self.class_encoder = DilatedConvEncoder(
            input_dims,
            [hidden_dims] * depth + [reduced_dims],
            output_dims,
            kernel_size=3
        )
        cores = torch.empty(classes, output_dims).to(torch.float)
        torch.nn.init.xavier_uniform_(cores, gain=torch.nn.init.calculate_gain('leaky_relu'))
        self.cores = torch.nn.Parameter(cores, requires_grad=False)
        self.encoder_list = torch.nn.ModuleList([DilatedConvEncoder(
            input_dims,
            [hidden_dims] * depth + [reduced_dims],
            output_dims,
            kernel_size=3
        ) for i in range(classes)])

    def forward(self, x, only_class=False):
        cores = torch.nn.functional.normalize(self.cores, dim=1)
        class_representation = torch.nn.functional.normalize(self.class_encoder(x), dim=1)

        if self.sim_fun == 'cosine':
            cates_logits = torch.mm(class_representation, cores.transpose(0,1))
            # cates_logits = torch.cat([torch.cosine_similarity(class_representation[i].reshape(1,-1), cores).reshape(1,-1) for i in range(class_representation.shape[0])])
        elif self.sim_fun == 'euclidean':
            cates_logits = torch.div(1, torch.cat([torch.nn.functional.pairwise_distance(class_representation[i], cores, p=2).reshape(-1,self.classes) for i in range(class_representation.shape[0])]))

        cates_logits = cates_logits / self.tau

        representation = torch.zeros((x.shape[0], self.output_dims), dtype=torch.float).to(cates_logits.device)
        
        if self.cate_fun == 'gumbel':
            cates = torch.nn.functional.gumbel_softmax(cates_logits, hard=True)
        elif self.cate_fun == 'softmax':
            cates = torch.nn.functional.softmax(cates_logits, dim=1)
        elif self.cate_fun == 'argmax':
            arg = torch.argmax(cates_logits, dim=-1)
            cates = torch.zeros((x.shape[0], self.classes), dtype=torch.float).to(cates_logits.device)
            cates[:, arg] = 1.0
        
        if not only_class:
            for k in range(self.classes):
                index = cates[:,k] > 0.001
                if sum(index) > 0:
                    representation[index] += self.encoder_list[k](x[index]) * cates[index,k].reshape(-1,1)

        return {
            'cates': cates,
            'repr': representation,
            'class_repr': class_representation
        }