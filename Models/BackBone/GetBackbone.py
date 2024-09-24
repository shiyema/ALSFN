from . import drn
from .ResNet import *
from .VGGNet import *
from .batchnorm import SynchronizedBatchNorm2d
__all__ = ['get_backbone']


def get_backbone(model_name='', pretrained=True, num_classes=None, **kwargs):
    if 'res' in model_name:
        model = get_resnet(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)

    elif 'vgg' in model_name:
        model = get_vgg(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif 'drn' in model_name:
        model = drn.drn_c_42(SynchronizedBatchNorm2d)
    else:
        raise NotImplementedError
    return model

