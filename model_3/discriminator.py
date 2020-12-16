import torch
from torch import nn
from torch.nn import functional as F


# ---
# Simple discriminator
# ---

class discriminator_1(nn.Module):
    def __init__(self, emb_dim , domain_dims):
        super(discriminator_1, self).__init__()
        self.num_domains = len(domain_dims)
        self.emb_list = nn.ModuleList(
            [nn.Embedding(domain_dims[i], emb_dim) for i in range(self.num_domains)]
        )
        self.K = int( (self.num_domains *(self.num_domains -1))//2)
        self.FC = nn.Linear(self.K, 1, bias=False)
        self.W = nn.parameter.Parameter(torch.ones([self.K,1]))
        self.emb_dim = emb_dim
        return


    def forward(self, x):
        x_split = torch.chunk(x, self.num_domains, dim=1)
        # self.normalize_embedding()
        comp_ij = []
        for i in range(self.num_domains):
            for j in range(i,self.num_domains):
                if i == j : continue
                a = self.emb_list[i](x_split[i].squeeze(1)) # Shape : [batch , emb_dim]
                b = self.emb_list[j](x_split[j].squeeze(1)) # Shape : [batch , emb_dim]
                _ij = F.cosine_similarity (a,b, dim=-1)
                comp_ij.append(_ij)

        comp_ij = torch.stack(comp_ij, dim=-1)  # Shape : [batch ,K ]
        res = torch.matmul( comp_ij, torch.exp(self.W))
        res = torch.sigmoid(res)
        return res
    
    def normalize_embedding(self):
        for emb in self.emb_list :
            norms = torch.norm(emb.weight, p=2, dim=1).data
            emb.weight.data = emb.weight.data.div(norms.view(emb.weight.shape[0],1).expand_as(emb.weight))


class discriminator_2(nn.Module):
    def __init__(self, emb_dim, domain_dims):
        super(discriminator_2,  self).__init__()
        self.num_domains = len(domain_dims)
        self.emb_list = nn.ModuleList(
            [nn.Embedding(domain_dims[i], emb_dim) for i in range(self.num_domains)]
        )
        self.K = int((self.num_domains * (self.num_domains - 1)) // 2)
        self.emb_dim = emb_dim
        return

    def forward(self, x):
        x_split = torch.chunk(x, self.num_domains, dim=1)
        comp_ij = []
        for i in range(self.num_domains):
            a = self.emb_list[i](x_split[i].squeeze(1))  # Shape : [batch , emb_dim]
            comp_ij.append(a)

        comp_ij = torch.sum(comp_ij, dim=-2)  # Shape : [batch ,K ]
        res = torch.norm(comp_ij,dim=-1)
        res = torch.tanh(res)
        return res

    def normalize_embedding(self):
        for emb in self.emb_list:
            norms = torch.norm(emb.weight, p=2, dim=1).data
            emb.weight.data = emb.weight.data.div(norms.view(emb.weight.shape[0], 1).expand_as(emb.weight))