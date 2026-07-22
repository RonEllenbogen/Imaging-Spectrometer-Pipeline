'''
This script plots a raw image. Using this as a validation step.
'''

# Imports

from pathlib import Path
import imageio.v3 as iio
import matplotlib.pyplot as plt

# Constants

# Classes

# Functions

def plot_image(filename: str):
    """
    Load and display a raw image from data/raw.

    Parameters
    ----------
    filename
        Name of the raw image file to be plotted
    """

    # Get image from file path
    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "data" / "raw" / filename

    image = iio.imread(image_path)

    # Plot image
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        image,
        cmap="viridis",
        #vmin=0, vmax=10, # Uncomment to change brightness limits to change contrast
        origin="upper",
        aspect="auto"
    )

    ax.set_title(filename)
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pixel intensity")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_image("test_image.tiff")