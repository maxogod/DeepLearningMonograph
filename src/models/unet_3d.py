import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Encoder
        self.c1 = DoubleConv(in_channels, 16, 0.1)
        self.p1 = nn.MaxPool3d(2)

        self.c2 = DoubleConv(16, 32, 0.1)
        self.p2 = nn.MaxPool3d(2)

        self.c3 = DoubleConv(32, 64, 0.2)
        self.p3 = nn.MaxPool3d(2)

        self.c4 = DoubleConv(64, 128, 0.2)
        self.p4 = nn.MaxPool3d(2)

        self.c5 = DoubleConv(128, 256, 0.3)

        # Decoder
        self.up6 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0)
        self.c6 = DoubleConv(256, 128, 0.2)

        self.up7 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
        self.c7 = DoubleConv(128, 64, 0.2)

        self.up8 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
        self.c8 = DoubleConv(64, 32, 0.1)

        self.up9 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0)
        self.c9 = DoubleConv(32, 16, 0.1)

        self.out = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        c5 = self.c5(p4)

        # Decoder
        u6 = self.up6(c5)
        # crop c4 to match u6 size
        c4_crop = self.center_crop(c4, u6.shape[2:])
        u6 = torch.cat([u6, c4_crop], dim=1)
        c6 = self.c6(u6)

        u7 = self.up7(c6)
        c3_crop = self.center_crop(c3, u7.shape[2:])
        u7 = torch.cat([u7, c3_crop], dim=1)
        c7 = self.c7(u7)

        u8 = self.up8(c7)
        c2_crop = self.center_crop(c2, u8.shape[2:])
        u8 = torch.cat([u8, c2_crop], dim=1)
        c8 = self.c8(u8)

        u9 = self.up9(c8)
        c1_crop = self.center_crop(c1, u9.shape[2:])
        u9 = torch.cat([u9, c1_crop], dim=1)
        c9 = self.c9(u9)

        return self.out(c9)

    @staticmethod
    def center_crop(layer, target_size):
        """Crop layer to target_size (D,H,W)"""
        _, _, d, h, w = layer.shape
        td, th, tw = target_size
        ds = (d - td) // 2
        hs = (h - th) // 2
        ws = (w - tw) // 2
        return layer[:, :, ds : ds + td, hs : hs + th, ws : ws + tw]


if __name__ == "__main__":
    from torchinfo import summary

    model = UNet3D(3, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(in_channels=3, num_classes=4).to(device)

    x = torch.randn(1, 3, 128, 128, 128).to(device)
    output = model(x)

    # loss = criterion(output, y)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    summary(model, input_size=(1, 3, 128, 128, 128))
