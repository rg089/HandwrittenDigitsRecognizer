from tensorflow.keras.utils import plot_model
import os


def visualize_dimensions(model, name, path=None, vertical=True):
    """
    Visualize the model architecture and input and output dimensions of each layer.
    :param model: The model object
    :param name: Name of the saved file. Can be given with extension, otherwise default is used.
    :param path: The location of the folder to save in. Default is plots/architectures.
    :param vertical: Whether orientation is vertical or horizontal.
    :return: Nothing.
    """
    if path and not os.path.exists(path):
        print("Provided path {} doesn't exist! Defaulting to {}...".format(os.path.abspath(path),
                                                                           os.path.abspath("..\\plots\\architectures")))
        path = None
    if not path:
        path = "..\\plots\\architectures"
    if "." not in name:
        name += ".jpg"
    loc = path + "\\" + name
    orientation = "TB"
    if not vertical:
        orientation = "LR"
    plot_model(model, loc, show_shapes=True, rankdir=orientation)
    print("Plot created successfully at {}".format(os.path.abspath(loc)))
