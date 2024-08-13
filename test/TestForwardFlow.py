import torch

from Utils.ImageUtils import imageToTensor, get_tensor_to_image_transform
from diffusion.forwardProcess import get_noisy_image


def getANoisyImageFromCleanImage():
    img_tensor = imageToTensor("/Users/shubhamsingh/Desktop/starry_night.png")
    noisy_image = get_noisy_image(img_tensor, torch.tensor([20]))
    noisy_image.show()

if __name__ == "__main__":
    getANoisyImageFromCleanImage()