import torch
import torch.nn as nn
from Utils import DataUtils
from model.model import Unet

import forwardProcess     # Importing this will calculate the alpha and all


def initiate_training(epochs = 1):
    image_size = 28
    channels = 1
    batch_size = 128
    timesteps = 300
    kill_switch = 0

    dataloader = DataUtils.prepare_data_loader(batch_size)
    loss_fn = nn.MSELoss()

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(epochs):
        for batch, _ in dataloader:
            kill_switch += 1
            t = torch.randint(0, timesteps, (batch.shape[0],)).long()  # The integers are sampled uniformly from the range [0, timesteps),
            noise = torch.randn_like(batch)  # Sampling noise from a gaussian distribution for every training example in batch
            noisy_images = forwardProcess.forward_process(batch, t, noise)

            predicted_noise = model(noisy_images, t)
            loss = loss_fn(predicted_noise, noise)

            print(f"Loss: {loss}")

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if kill_switch > 5:
                break

if __name__ == "__main__":
    initiate_training()