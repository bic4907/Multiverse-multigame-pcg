import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_rate)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, out_ch // 4, 1),
                nn.ReLU(),
                nn.Conv2d(out_ch // 4, out_ch, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.dropout(self.conv2(h))
        if self.use_se:
            h = h * self.se(h)
        return self.act(h + self.skip(x))


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, drop_rate=0.1, use_se=False):

        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_rate)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, out_ch // 4, 1),
                nn.ReLU(),
                nn.Conv2d(out_ch // 4, out_ch, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.dropout(self.conv2(h))
        if self.use_se:
            h = h * self.se(h)
        return self.act(h + self.skip(x))

