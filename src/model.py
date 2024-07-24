import timm
import torch.nn as nn
from torchvision.models import (ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, resnet50, resnet101,
                                resnet152)


class CustomResNet(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', fc_layer=512, dropout_rate=0.5, freeze_pretrained=True):
        super(CustomResNet, self).__init__()
        if model_name == "resnet50":
            self.resnet = timm.create_model('resnet50.a1_in1k', pretrained=True)
        # if model_name == "resnet50":
        #     self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "resnet101":
            self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)        
        elif model_name == "resnet152":
            self.resnet = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("model_name should be 'resnet50', 'resnet101', or 'resnet152'")
        
        if freeze_pretrained:   
            for param in self.resnet.parameters():
                param.requires_grad = False
            for module in self.resnet.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = True

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, fc_layer),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_layer, num_classes),
            nn.Softmax(dim=1)
            )
        
        for param in self.resnet.fc.parameters():
            param.requires_grad = True



    def forward(self, x):
        return self.resnet(x)