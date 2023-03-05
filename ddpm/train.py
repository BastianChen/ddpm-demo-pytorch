import torch
import os
import argparse
from net import UNetModel
from diffusion import GaussianDiffusion
from torchvision import datasets, transforms


def run(args):
    batch_size = args.batch_size
    epochs = args.epochs
    timesteps = args.timesteps
    datasets_path = args.datasets_path
    datasets_type = args.datasets_type

    if not os.path.exists("models"):
        os.mkdir("models")

    if datasets_type:
        dataset = datasets.CIFAR10(
            root=datasets_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        lr = 2e-4
        in_channels = 3
        save_path = f"models/cifar10-{timesteps}-{epochs}-{lr}.pth"
    else:
        dataset = datasets.MNIST(root=datasets_path, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]))
        lr = 5e-4
        in_channels = 1
        save_path = f"models/mnist-{timesteps}-{epochs}-{lr}.pth"
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define model and diffusion
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = UNetModel(
        in_channels=in_channels,
        model_channels=96,
        out_channels=in_channels,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

    # train
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(train_loader):
            batch_size = images.shape[0]
            images = images.to(device)

            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = gaussian_diffusion.train_losses(model, images, t)

            if step % 200 == 0:
                print("Loss:", loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('-d', '--datasets_type', default=0, type=int, help="datasets type,0:MNISI,1:cifar-10")
    parser.add_argument('-e', '--epochs', default=20, type=int)
    parser.add_argument('-t', '--timesteps', default=500, type=int, help="timesteps")
    parser.add_argument('-dp', '--datasets_path', default="../datasets", type=str, help="path of Datasets")

    args = parser.parse_args()

    run(args)
