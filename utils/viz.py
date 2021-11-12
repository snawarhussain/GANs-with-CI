from torchvision.utils import make_grid
from matplotlib import pyplot as plt


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), show=False, save=False):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / \
                   2  # makes black pixels (0 value) gray pixels(0.5 values)
    # while white pixels (1 values) remain unchanged
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
    if save:
        plt.imsave("./graph.png", image_grid.permute(1, 2, 0).squeeze().numpy())


def show_real_fake(real, fake):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    show_tensor_images(real)
    plt.subplot(1, 2, 2)
    show_tensor_images(fake)
    plt.show()
