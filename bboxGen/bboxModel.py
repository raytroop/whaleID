import torch
import torch.nn as nn
import torch.nn.functional as F


LENGTH = 384

class bbox_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 9), padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),    # 384

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),    # 192

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),    # 96

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),    # 48

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),    # 24

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),    # 12

        )
        lgth = 64 * LENGTH // (2**5)
        self.ho = nn.Linear(in_features=lgth, out_features=32)
        self.bho = nn.BatchNorm1d(num_features=32)
        self.vt = nn.Linear(in_features=lgth, out_features=32)
        self.bvt = nn.BatchNorm1d(num_features=32)
        self.out = nn.Linear(in_features=32*2, out_features=4)  # (x0, y0, x1, y1)

    def forward(self, img):
        feats = self.body(img)
        assert feats.shape[2] == feats.shape[3]

        h = F.max_pool2d(feats, kernel_size=(1, feats.shape[2]))
        h = h.view(h.shape[0], -1)
        ho = F.relu(self.ho(h))
        ho = self.bho(ho)

        v = F.max_pool2d(feats, kernel_size=(feats.shape[2], 1))
        v = v.view(v.shape[0], -1)
        vo = F.relu(self.vt(v))
        vo = self.bvt(vo)

        o = torch.cat((ho, vo), 1)
        out = self.out(o)

        return out


if __name__ == '__main__':
    img = torch.randn(10, 1, LENGTH, LENGTH)
    model = bbox_model()
    out = model(img)
    print(out.shape)
