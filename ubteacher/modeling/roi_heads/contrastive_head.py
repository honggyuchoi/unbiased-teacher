# From https://github.com/MegviiDetection/FSCE/blob/main/fsdet/modeling/contrastive_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in=1024, feat_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in), # 1024, 1024
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim), # 1024, 128 
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized

# Two fc layers with relu
class Projector(nn.Module):
    def __init__(self, dim_in=1024, hidden_dim=2048, feat_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, hidden_dim, bias=False), # 1024, 1024
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim, bias=False), # 1024, 128 
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        assert feat.dim() == 2, "feat dimension is not 2"
        return feat
        
# Two fc layers with batchnorm and relu
# projector and predictor architeture is same as mocov3
class Projector_bn(nn.Module):
    def __init__(self, dim_in=1024, hidden_dim=2048, feat_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, hidden_dim, bias=False), # 1024, 1024
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim, bias=False), # 1024, 128 
            nn.BatchNorm1d(feat_dim, affine=False),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        assert feat.dim() == 2, "feat dimension is not 2"
        return feat

# Two fc layers with relu
class Predictor(nn.Module):
    def __init__(self, dim_in=128, hidden_dim=2048, feat_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, hidden_dim, bias=False), # 128, 1024
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim, bias=False), # 1024, 128 
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        assert feat.dim() == 2, "feat dimension is not 2"
        return feat

# Two fc layers with batchnorm and relu
class Predictor_bn(nn.Module):
    def __init__(self, dim_in=128, hidden_dim=2048, feat_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, hidden_dim, bias=False), # 1024, 1024
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim, bias=False), # 1024, 128 
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        assert feat.dim() == 2, "feat dimension is not 2"
        return feat



# # Two fc layers with batchnorm and relu
# class MLP_BN(nn.Module):
#     def __init__(self, in_dim, inner_dim=4096, out_dim=256):
#         super(MLP, self).__init__()

#         self.linear1 = nn.Linear(in_dim, inner_dim)
#         self.bn1 = nn.BatchNorm1d(inner_dim)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.linear2 = nn.Linear(inner_dim, out_dim)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = x.unsqueeze(-1)
#         x = self.bn1(x)
#         x = x.squeeze(-1)
#         x = self.relu1(x)

#         x = self.linear2(x)

#         return x

# def Proj_Head(in_dim=2048, inner_dim=4096, out_dim=256):
#     return MLP(in_dim, inner_dim, out_dim)


# def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
#     return MLP(in_dim, inner_dim, out_dim)