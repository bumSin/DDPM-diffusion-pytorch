import torch

from Utils.ImageUtils import get_tensor_to_image_transform
from Utils.ScheduleUtils import linear_beta_schedule

# pre calculate values of beta, underroot alpha_bar and 1-alpha_bar for all timesteps

timesteps = 500
betas = linear_beta_schedule(timesteps=timesteps)

alphas = 1. - betas
alpha_bars = torch.cumprod(alphas, axis=0)  # alpha_bar  = alpha1 * alpha2 * ... * alphat
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# forward diffusion (using the nice property)
def sample_a_new_image_tensor(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alpha_bars, t, x_start.shape)  # Read as, extract values at index t from sqrt_alpha_bars and reshape it to match x_start.shape
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alpha_bars, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t):
    # sample a new noisy image
    x_noisy = sample_a_new_image_tensor(x_start, t=t)  # This is a tensor

    # turn back into PIL image
    reverse_transform = get_tensor_to_image_transform()
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image

