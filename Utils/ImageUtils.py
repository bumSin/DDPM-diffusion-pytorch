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

def saveImage(image, image_size, epoch, step, results_folder):
    # Convert the image tensor to a format suitable for saving
    # Extract the first image (batch=1), channel=1 and reshape
    image_array = image[0][0].reshape(image_size, image_size)

    # Convert to the range [0, 255] for saving as an image
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
    image_array = image_array.astype('uint8')

    # Create a PIL Image from the NumPy array and save as PNG
    image = Image.fromarray(image_array)

    # Define the full path including the directory and file name
    save_path = os.path.join(results_folder, f"epoch_{epoch}_step_{step}.png")

    # Save the image as a PNG file in the specified directory
    image.save(save_path)