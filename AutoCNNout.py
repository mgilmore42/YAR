import matplotlib.pyplot as plt
import numpy as np
from AutoCNN import Autoencoder_CNN, ImageDataset, transform
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch


# Load the trained model
model = Autoencoder_CNN()
model.load_state_dict(torch.load("autoencoder_cnn_weights.pt"))

# Load a single image from the dataset
image_dir = "phoday2images"
dataset = ImageDataset(image_dir, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_iter = iter(loader)
image, _ = next(data_iter)

# Pass the image through the autoencoder
output = model(image)

# Convert the tensors to numpy arrays and rescale the pixel values to [0, 1]
image_np = np.transpose(image.squeeze().numpy(), (1, 2, 0))
output_np = np.transpose(output.squeeze().detach().numpy(), (1, 2, 0))
image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
output_np = (output_np - np.min(output_np)) / (np.max(output_np) - np.min(output_np))

# Display the original and reconstructed images
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(image_np)
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(output_np)
ax[1].set_title("Reconstructed Image")
ax[1].axis("off")
plt.show()
