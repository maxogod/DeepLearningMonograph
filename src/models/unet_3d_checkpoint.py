from typing import cast

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class UNet3DCheckpoint(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base_ch=16):
        super().__init__()

        # Encoder
        self.enc0 = self._conv_block(in_channels, base_ch)
        self.enc1 = self._conv_block(base_ch, base_ch * 2)
        self.enc2 = self._conv_block(base_ch * 2, base_ch * 4)
        self.enc3 = self._conv_block(base_ch * 4, base_ch * 8)
        self.enc4 = self._conv_block(base_ch * 8, base_ch * 16)

        # Decoder
        self.up1 = self._up_block(base_ch * 16, base_ch * 8)
        self.up2 = self._up_block(base_ch * 8 * 2, base_ch * 4)
        self.up3 = self._up_block(base_ch * 4 * 2, base_ch * 2)
        self.up4 = self._up_block(base_ch * 2 * 2, base_ch)

        self.final = nn.Conv3d(base_ch * 2, num_classes, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        e0 = cast(torch.Tensor, cp.checkpoint(self.enc0, x, use_reentrant=False))
        e1 = cast(
            torch.Tensor,
            cp.checkpoint(
                lambda t: self.enc1(nn.functional.max_pool3d(t, 2)),
                e0,
                use_reentrant=False,
            ),
        )
        e2 = cast(
            torch.Tensor,
            cp.checkpoint(
                lambda t: self.enc2(nn.functional.max_pool3d(t, 2)),
                e1,
                use_reentrant=False,
            ),
        )
        e3 = cast(
            torch.Tensor,
            cp.checkpoint(
                lambda t: self.enc3(nn.functional.max_pool3d(t, 2)),
                e2,
                use_reentrant=False,
            ),
        )
        e4 = cast(
            torch.Tensor,
            cp.checkpoint(
                lambda t: self.enc4(nn.functional.max_pool3d(t, 2)),
                e3,
                use_reentrant=False,
            ),
        )

        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d4 = self.up4(d3)
        d4 = torch.cat([d4, e0], dim=1)

        out = self.final(d4)
        return torch.sigmoid(out)


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3DCheckpoint(in_channels=3, num_classes=4).to(device)

    x = torch.randn(1, 3, 128, 128, 128).to(device)
    output = model(x)

    # loss = criterion(output, y)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    summary(model, input_size=(1, 3, 128, 128, 128))
