import torch

from Utils.ImageUtils import imageToTensor
from diffusion.forwardProcess import get_noisy_image, forward_process


def getANoisyImageFromCleanImage():
    img_tensor = imageToTensor("/Users/shubhamsingh/Desktop/starry_night.png")
    noisy_image = get_noisy_image(img_tensor, torch.tensor([20]))
    noisy_image.show()

def forwardFlowBatchTest():
    # This tests the call which will be made from the training loop on a batch of images
    x_start_batch = torch.randn(2, 1, 28, 28)
    t = torch.randn(2, ).long()

    sampled_tensor = forward_process(x_start_batch, t)

    # Assert that the shape of the sampled tensor matches the shape of x_start_batch
    assert sampled_tensor.shape == x_start_batch.shape, \
        f"Shape mismatch: expected {x_start_batch.shape}, but got {sampled_tensor.shape}"
    print(f"sampled_tensor.shape: {sampled_tensor.shape}")

if __name__ == "__main__":
    getANoisyImageFromCleanImage()
    forwardFlowBatchTest()