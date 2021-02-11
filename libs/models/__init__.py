import os
from typing import Dict

import torch
import torch.nn as nn

from .DCGAN import DCDiscriminator, DCGenerator
from .BigGAN import BigEncoder, BigDiscriminator, BigGenerator

__all__ = ["get_model"]

model_names = ["DCGAN", "BIGGAN"]


def get_model(name: str, z_dim: int = 20, image_size: int = 64) -> Dict[str, nn.Module]:
    name = name.upper()
    if name not in model_names:
        raise ValueError(
            """There is no model appropriate to your choice.
            You have to choose DCGAN or BigGAN as a model.
        """
        )

    print("{} will be used as a model.".format(name))
    
    if name == "DCGAN":
        G = DCGenerator(z_dim, image_size)
        D = DCDiscriminator(z_dim, image_size)
        G.apply(DCweights_init)
        D.apply(DCweights_init)
        model = {
            "G": G,
            "D": D,
        }

    elif name == "BIGGAN":
        G = BigGenerator(z_dim)
        D = BigDiscriminator(z_dim)
        E = BigEncoder(z_dim)
        G.apply(Bigweights_init)
        D.apply(Bigweights_init)
        E.apply(Bigweights_init)
        model = {
            "G": G,
            "D": D,
            "E": E,
        }

    return model


def DCweights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def Bigweights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)
