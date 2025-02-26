import datasets
import numpy as np
from PIL import Image

# Load CIFAR-10 dataset
test_dataset = datasets.load_dataset("cifar10")["test"]

# Define noise functions
def add_gaussian_noise(image, intensity):
    img = np.array(image)
    noise = np.random.normal(0, intensity, img.shape)
    noisy_img = img + noise
    return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))

def add_speckle_noise(image, intensity):
    img = np.array(image)
    noise = np.random.gamma(1, intensity, img.shape)
    noisy_img = img * noise
    return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))

def add_label_noise(label, intensity):
    if np.random.rand() < intensity:
        return np.random.randint(0, 10)
    return label

# Create noisy versions of the test set
noise_types = ["gaussian", "speckle", "label"]
intensities = [0.0, 0.1, 0.2, 0.5, 0.8]

for noise_type in noise_types:
    for intensity in intensities:
        print(f"Processing {noise_type} noise with intensity {intensity}")
        
        if noise_type == "gaussian":
            new_dataset = test_dataset.map(
                lambda x: {"img": add_gaussian_noise(x["img"], intensity), "label": x["label"]},
                batched=False
            )
        elif noise_type == "speckle":
            new_dataset = test_dataset.map(
                lambda x: {"img": add_speckle_noise(x["img"], intensity), "label": x["label"]},
                batched=False
            )
        elif noise_type == "label":
            new_dataset = test_dataset.map(
                lambda x: {"img": x["img"], "label": add_label_noise(x["label"], intensity)},
                batched=False
            )
        
        print(f"Done processing {noise_type} noise with intensity {intensity}")