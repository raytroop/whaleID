import torch
import torch.nn as nn
import torch.nn.functional as F

class headCONV(nn.Module):
    def __init__(self,in_features=512,  mid=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=mid, kernel_size=(4, 1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(mid, 1))
        self.fc = nn.Linear(in_features=in_features, out_features=1)
    def forward(self, featsA, featsB):
        """
        Args:
            featsA: (None, in_features)
            featsB: (None, in_features)
        """
        x1 = featsA * featsB
        x2 = featsA + featsB
        x3 = torch.abs(featsA - featsB)
        x4 = torch.pow(x3, 2)
        x = torch.stack([x1, x2, x3, x4], dim=1)    # (?, 4, in_features)
        x = torch.unsqueeze(x, dim=1)  # (?, 1, 4, in_features)
        x = self.conv1(x)   # (?, mid, 1, in_features)
        x = F.relu(x)
        x = torch.squeeze(x, dim=2)     # (?, mid, in_features)
        x = torch.unsqueeze(x, dim=1)   # (?, 1, mid, in_features)
        x = self.conv2(x)   # (?, 1, 1, in_features)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)   # (?, in_features)
        x = self.fc(x)
        return torch.sigmoid(x)

class headFC(nn.Module):
    def __init__(self, in_features=512, mid=[128, 32]):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(num_features=in_features)
        self.fc1 = nn.Linear(in_features=in_features, out_features=mid[0])
        self.bn1 = nn.BatchNorm1d(num_features=mid[0])
        self.fc2 = nn.Linear(in_features=mid[0], out_features=mid[1])
        self.bn2 = nn.BatchNorm1d(num_features=mid[1])
        self.out = nn.Linear(in_features=mid[1], out_features=1)
    def forward(self, featsA, featsB):
        x = torch.pow(featsA - featsB, 2)
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.out(x)
        return F.sigmoid(x)


#https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


if __name__ == '__main__':
    model = headModel()
    featsA = torch.rand(20, 512)
    featsB = torch.rand(20, 512)
    out = model(featsA, featsB)
    print(out)
