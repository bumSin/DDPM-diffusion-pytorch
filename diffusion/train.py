from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from Utils import DataUtils
from Utils.ImageUtils import saveImage
from diffusion.sampling import sample
from model.model import Unet

import forwardProcess     # Importing this will calculate the alpha and all


def initiate_training(epochs = 1):
    image_size = 28
    channels = 1
    batch_size = 128
    timesteps = 300
    save_and_sample_every = 1000
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)

    dataloader = DataUtils.prepare_data_loader(batch_size)
    loss_fn = nn.MSELoss()

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for step, (batch, _) in enumerate(dataloader):
            t = torch.randint(0, timesteps, (batch.shape[0],)).long()  # The integers are sampled uniformly from the range [0, timesteps),
            noise = torch.randn_like(batch)  # Sampling noise from a gaussian distribution for every training example in batch
            noisy_images = forwardProcess.forward_process(batch, t, noise)

            predicted_noise = model(noisy_images, t)
            loss = loss_fn(predicted_noise, noise)

            print(f"Loss: {loss}")

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if step != 0 and step % save_and_sample_every == 0:
                sample_im = sample(model, image_size, batch_size=1, channels=1, timesteps=timesteps)
                saveImage(sample_im, image_size, epoch, step, results_folder)

    # sample_im = sample(model, image_size, batch_size=1, channels=1, timesteps=timesteps)
    # plt.imshow(sample_im[0][0].reshape(image_size, image_size, channels), cmap="gray") # plt.imshow just finishes drawing a picture instead of printing it.
    # plt.show()  # If you want to print the picture, you need to add plt.show.


if __name__ == "__main__":
    initiate_training()