import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from libs.anomaly import Anomaly_score
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.graph import make_image
from libs.mean_std import get_mean, get_std
from libs.models import get_model
from libs.transformer import ImageTransform

random_seed = 1234


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
        train GAN for object detection with Mnist Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    config = get_config(args.config)

    result_path = os.path.dirname(args.config)

    device = get_device(allow_only_gpu=True)

    model = get_model(name=config.model, z_dim=config.z_dim, image_size=config.size)
    for k, v in model.items():
        state_dict = torch.load(os.path.join(result_path, "final_model_%s.prm" % k))

        v.load_state_dict(state_dict)
        v.to(device)
        v.eval()

    train_loader = get_dataloader(
        csv_file=config.train_csv,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=ImageTransform(mean=get_mean(), std=get_std()),
    )

    batch_iterator = iter(train_loader)
    imgs = next(batch_iterator)  

    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i+1)
        plt.imshow(imgs[i][0].cpu().detach().numpy(), 'gray')
    plt.savefig(os.path.join(result_path, "sample.png"))

    x = imgs[0:5]
    x = x.to(device)

    z = torch.randn(5, 20).to(device)
    z = z.view(z.size(0), z.size(1), 1, 1)

    z.requires_grad = True
    z_optimizer = torch.optim.Adam([z], lr=1e-3)

    for epoch in range(5000+1):
        fake_img = model["G"](z)
        loss, _, _ = Anomaly_score(x, fake_img, model["D"], Lambda=0.1)

        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()

        if epoch % 1000 == 0:
            print('epoch {} || loss_total:{:.0f} '.format(epoch, loss.item()))

    model["G"].eval()
    fake_img = model["G"](z)

    loss, loss_each, residual_loss_each = Anomaly_score(x, fake_img, model["D"], Lambda=0.1)

    loss_each = loss_each.cpu().detach().numpy()
    print("total loss：", np.round(loss_each, 0))

    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i+1)
        plt.imshow(imgs[i][0].cpu().detach().numpy(), 'gray')

        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_img[i][0].cpu().detach().numpy(), 'gray')
    plt.savefig(os.path.join(result_path, "anorm.png"))

    print("Done")


if __name__ == "__main__":
    main()
