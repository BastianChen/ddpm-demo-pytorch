import torch
import os
import argparse
import matplotlib.pyplot as plt
from net import UNetModel
from diffusion import GaussianDiffusion


def inference(args):
    batch_size = args.batch_size
    timesteps = args.timesteps
    datasets_type = args.datasets_type

    if datasets_type:
        in_channels = 3
        image_size = 32
        save_image_name = "cifar10"
    else:
        in_channels = 1
        image_size = 28
        save_image_name = "mnist"

    # define model and diffusion
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = UNetModel(
        in_channels=in_channels,
        model_channels=96,
        out_channels=in_channels,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )

    map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
    model.to(device)
    model.load_state_dict((torch.load(args.pth_path, map_location=map_location)))
    model.eval()

    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

    label = torch.randint(0, 10, (batch_size,)).to(device)
    label_cifar = ['plane ', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    generated_images = gaussian_diffusion.sample(model, label, image_size, batch_size=batch_size, channels=in_channels)
    # generated_images: [timesteps, batch_size=64, channels=1, height=28, width=28]

    # generate new images
    if not os.path.exists("photos"):
        os.mkdir("photos")
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(8, 8)

    if datasets_type:
        imgs = generated_images[-1].reshape(8, 8, 3, 32, 32)
    else:
        imgs = generated_images[-1].reshape(8, 8, 28, 28)
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = imgs[n_row, n_col]
            if datasets_type:
                img = img.swapaxes(0, 1)
                img = img.swapaxes(1, 2)
                f_ax.imshow(((img + 1.0) * 255 / 2) / 255)
            else:
                f_ax.imshow((img + 1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
            plt.title(
                f"{label_cifar[label[n_row * 8 + n_col]] if datasets_type else label[n_row * 8 + n_col]}")
    f = plt.gcf()  # 获取当前图像
    f.savefig(f'photos/classifier_free_{save_image_name}_1.png')
    f.clear()  # 释放内存

    # show the denoise steps
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    rows = 12  # len(y)
    gs = fig.add_gridspec(rows, 16)
    for n_row in range(rows):
        for n_col in range(16):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
            img = generated_images[t_idx][n_row]
            if datasets_type:
                img = img.swapaxes(0, 1)
                img = img.swapaxes(1, 2)
                f_ax.imshow(((img + 1.0) * 255 / 2) / 255)
            else:
                img = img[0]
                f_ax.imshow((img + 1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
            plt.title(f"{label_cifar[label[n_row]] if datasets_type else label[n_row]}")

    f = plt.gcf()  # 获取当前图像
    f.savefig(f'photos/classifier_free_{save_image_name}_2.png')
    f.clear()  # 释放内存


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('-d', '--datasets_type', default=0, type=int, help="datasets type,0:MNISI,1:cifar-10")
    parser.add_argument('-t', '--timesteps', default=1000, type=int, help="timesteps")
    parser.add_argument('-p', '--pth_path', default="models/cifar10-1000-100-0.0002.pth", type=str,
                        help="path of the pth file")

    args = parser.parse_args()
    print(args)
    inference(args)
