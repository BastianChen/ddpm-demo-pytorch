import torch
import os
import argparse
from net import UNetModel
from diffusion import GaussianDiffusion
from torchvision.utils import save_image


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

    # map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
    model.to(device)
    # model.load_state_dict((torch.load(args.pth_path, map_location=map_location)))
    model.load_state_dict((torch.load(args.pth_path)))
    model.eval()

    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    generated_images = gaussian_diffusion.sample(model, image_size, batch_size=batch_size, channels=in_channels)

    # generate new images
    if not os.path.exists("photos"):
        os.mkdir("photos")
    imgs = generated_images[-1].reshape(64, in_channels, image_size, image_size)
    img = torch.tensor(imgs)
    save_image(img, f'photos/{save_image_name}_1.png', 8, normalize=True, scale_each=True)

    imgs_time = []
    for n_row in range(16):
        for n_col in range(16):
            t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
            img = torch.tensor(generated_images[t_idx][n_row].reshape(in_channels, image_size, image_size))
            imgs_time.append(img)

    imgs = torch.stack(imgs_time).reshape(-1, in_channels, image_size, image_size)
    save_image(imgs, f'photos/{save_image_name}_2.png', 16, normalize=True, scale_each=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('-d', '--datasets_type', default=0, type=int, help="datasets type,0:MNISI,1:cifar-10")
    parser.add_argument('-t', '--timesteps', default=1000, type=int, help="timesteps")
    parser.add_argument('-p', '--pth_path', default="models/cifar10-1000-100-0.0002.pth", type=str, help="path of pth file")

    args = parser.parse_args()
    print(args)
    inference(args)
