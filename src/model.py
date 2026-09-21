import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class MultiTaskUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x):

        features = self.unet.encoder(x)

        decoder_output = self.unet.decoder(features)

        segmentation_output = self.unet.segmentation_head(
            decoder_output
        )

        deepest_feature = features[-1]

        classification_output = self.classifier(
            deepest_feature
        )

        return segmentation_output, classification_output
