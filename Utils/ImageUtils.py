import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms, Compose, Lambda, ToPILImage


def imageToTensor(img_path):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')

    img = Image.open(img_path)
    img = img.convert("RGB")  # remove alpha channel, if any

    # preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Normalize with the VGG16 mean and std
    ])

    # apply the preprocessing to image
    image_tensor = preprocess(img)
    return image_tensor


def tensor_to_image(img_tensor):
    # Assuming the tensor is in the format [C, H, W]
    img_tensor = img_tensor.squeeze(0)
    # Convert tensor to numpy array
    image_np = img_tensor.numpy()

    # Normalize the numpy array if necessary (assuming values are in [0, 1])
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # Display the image using matplotlib
    plt.imshow(image_np.transpose(1, 2, 0))  # Matplotlib expects channels-last format
    plt.axis('off')  # Turn off axis labels
    plt.show()

def get_tensor_to_image_transform():
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    return reverse_transform