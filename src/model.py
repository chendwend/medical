import timm
import torch.nn as nn
from torchvision.models import (ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, resnet50, resnet101,
                                resnet152)


class CustomResNet(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', fc_layer=512, dropout_rate=0.5, freeze_blocks=4):
        super(CustomResNet, self).__init__()
        # if model_name == "resnet50":
            # self.model = timm.create_model('resnet50.a1_in1k', pretrained=True)
        if model_name == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "resnet101":
            self.model = resnet101(weights=ResNet101_Weights.DEFAULT)        
        elif model_name == "resnet152":
            self.model = resnet152(weights=ResNet152_Weights.DEFAULT)
        elif model_name in ["resnetv2_50x1_bit", "resnetv2_101x1_bit", "resnetv2_152x2_bit"]:
            self.model = timm.create_model(model_name, pretrained=True)
        else:
            raise ValueError("Uknown model name.")
        
        # adapt first layer for grayscale images
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._freeze_layers(freeze_blocks)
        self._unfreeze_batchnorm_layers()

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, fc_layer),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_layer, num_classes),
            nn.Softmax(dim=1)
            )
        
        for param in self.model.fc.parameters():
            param.requires_grad = True


    def _freeze_layers(self, freeze_blocks):
        """Freeze the first N blocks, except for BatchNorm layers."""
        
        for block_count, (name, child) in enumerate(self.model.named_children()):
            if block_count < freeze_blocks:
                for param in child.parameters():
                    param.required_grad = False

    def _unfreeze_batchnorm_layers(self):
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True  


    def forward(self, x):
        return self.model(x)