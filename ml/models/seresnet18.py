from torch.utils import model_zoo
from torch import nn
from collections import OrderedDict


def _cfg(url="", **kwargs):
    return {
        "url": url, "num_classes": 1000, "input_size": (3, 224, 224), "pool_size": (7, 7),
        "crop_pct": 0.875, "interpolation": "bilinear",
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "first_conv": "layer0.conv1", "classifier": "last_linear",
        **kwargs
    }


default_cfg = {
    "seresnet18": _cfg(url="https://www.dropbox.com/s/3o3nd8mfhxod7rq/seresnet18-4bb0ce65.pth?dl=1", interpolation="bicubic")
}


class AdaptivePool2d(nn.Module):
    """

    Selectable global pooling layer with dynamic input kernel size

    """
    def __init__(self, output_size=1, pool_type="avg"):
        super(AdaptivePool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)

    def feat_mult(self):
        return 1

    def __repr__(self):
        return self.__class__.__name__ + " (" \
               + "output_size=" + str(self.output_size) \
               + ", pool_type=" + self.pool_type + ")"


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    def __init__(self, block=SEResNetBlock, layers=[2, 2, 2, 2], groups=1, reduction=16,
                 in_channels=3, inplanes=64, downsample_kernel_size=1,
                 downsample_padding=0, num_classes=1000, global_pool="avg"):
        super(SEResNet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes

        layer0_modules = [("conv1", nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                          ("bn1", nn.BatchNorm2d(inplanes)), ("relu1", nn.ReLU(inplace=True)),
                          ('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True))]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = AdaptivePool2d(pool_type=global_pool)
        self.num_features = 512 * block.expansion
        self.last_linear = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            self._weight_init(m)

    def _weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(
            self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        del self.last_linear
        if num_classes:
            self.last_linear = nn.Linear(self.num_features, num_classes)
        else:
            self.last_linear = None

    def forward_features(self, x, pool=True):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if pool:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        return x

    def logits(self, x):
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.logits(x)
        return x


def load_pretrained(model, default_cfg, num_classes=1000, in_channels=3, filter_fn=None):
    state_dict = model_zoo.load_url(default_cfg['url'])

    if in_channels == 1:
        conv1_name = default_cfg["first_conv"]
        print("Converting first conv (%s) from 3 to 1 channel" % conv1_name)
        conv1_weight = state_dict[conv1_name + ".weight"]
        state_dict[conv1_name + ".weight"] = conv1_weight.sum(dim=1, keepdim=True)
    elif in_channels != 3:
        raise AssertionError("Invalid in_channels for pretrained weights")

    strict = True
    classifier_name = default_cfg["classifier"]

    if num_classes != default_cfg["num_classes"]:
        del state_dict[classifier_name + ".weight"]
        del state_dict[classifier_name + ".bias"]
        strict = False

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)


def make_seresnet18(num_classes=1000, in_channels=3, pretrained=True, **kwargs):
    cfg = default_cfg["seresnet18"]
    model = SEResNet(SEResNetBlock, [2, 2, 2, 2], groups=1, reduction=16,
                     inplanes=64,
                     downsample_kernel_size=1, downsample_padding=0,
                     num_classes=num_classes, in_channels=in_channels, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        load_pretrained(model, cfg, num_classes, in_channels)
    return model


import torch
import time

model = make_seresnet18().to("cuda")
# print(sum(p.numel() for p in model.parameters()))

while True:
    inp = torch.rand(1, 3, 224, 224).to("cuda")
    start = time.time()
    print(model(inp).size())
    stop = time.time() - start
    print(stop)

