import os, sys
import numpy as np
import cv2
import argparse
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get_args(dataset=False, save_model=False, save_plot=False, save_json=False, save_weights=False, single_image=False,
             pretrained_model=False, **kwargs):
    ap = argparse.ArgumentParser()
    if dataset:
        help_ = "path to input dataset"
        if "dataset" in kwargs:
            help_ = kwargs["dataset"]
        ap.add_argument("-d", "--dataset", required=True, help=help_)
    if save_model:
        help_ = "path to output model"
        if "model" in kwargs:
            help_ = kwargs["model"]
        ap.add_argument("-m", "--model", type=str, default="inception", help=help_)
    elif pretrained_model:
        help_ = "name of pretrained model to use"
        if "model" in kwargs:
            help_ = kwargs["model"]
        ap.add_argument("-m", "--model", required=True, help=help_)
    if save_plot:
        help_ = "path to output plot"
        if "plot" in kwargs:
            help_ = kwargs["plot"]
        ap.add_argument("-p", "--plot", required=True, help=help_)
    if save_json:
        help_ = "path to output json"
        if "json" in kwargs:
            help_ = kwargs["json"]
        ap.add_argument("-j", "--json", required=True, help=help_)
    if save_weights:
        help_ = "path to weights directory"
        if "weights" in kwargs:
            help_ = kwargs["weights"]
        ap.add_argument("-w", "--weights", required=True, help=help_)
    if single_image:
        help_ = "path to the input image"
        if "image" in kwargs:
            help_ = kwargs["image"]
        ap.add_argument("-i", "--image", required=True, help=help_)
    args = vars(ap.parse_args())
    return args


def normalize(*args):
    ans = []
    for i in args:
        ans.append(i.astype("float") / 255.0)
    if len(args) == 1:
        return ans[0]
    return ans


def encodeY(*args, ohe=True):
    if ohe:
        encoder = LabelBinarizer()
    else:
        encoder = LabelEncoder()
    ans = [encoder, encoder.fit_transform(args[0])]
    for i in range(1, len(args)):
        ans.append(encoder.transform(args[i]))
    return ans


def addDimension(*args, after=True):
    ans = []
    if after:
        for i in args:
            ans.append(i[..., np.newaxis])
    else:
        for i in args:
            ans.append(i[np.newaxis, ...])
    if len(args) == 1:
        return ans[0]
    return ans


def indexOfFirstString(l):
    for i in range(len(l)):
        if type(l[i]) == str:
            return i
    return len(l)


def showImage(*imgs, together=False, **kwnames):
    i = indexOfFirstString(imgs)
    images = imgs[:i]
    names = list(imgs[i:])
    if not together:
        if len(images) > len(names):
            if len(kwnames) != 0:
                names.extend(list(kwnames.values())[:len(images) - len(names)])
            names.extend([f"Image{i}" for i in range(len(names) + 1, len(images) + 1)])
        for i in range(len(images)):
            cv2.imshow(names[i], images[i])
    else:
        if len(names) != 0:
            name = names[0]
        elif "name" in kwnames:
            name = kwnames["name"]
        elif len(kwnames) != 0:
            name = list(kwnames.values())[0]
        else:
            name = "Image"
        cv2.imshow(name, np.hstack(images))
    cv2.waitKey(0)
    cv2.destroyAllWindows()







def resize(img, width=None, height=None, inter=cv2.INTER_CUBIC, fx=1, fy=1, padSame=False):
    """
    Returns the resized image.
    :param img: cv2 image object
    :param width: The fixed width of the result
    :param height: The fixed height of the result
    :param inter: Interpolation Method
    :param fx: Ratio to scale width
    :param fy: Ratio to scale height
    :param padSame: If border is same or 0.
    :return: cv2 Image
    """
    h, w = img.shape[:2]

    if not width and not height:
        wn = int(w * fx)
        hn = int(h * fy)
        dim = (wn, hn)

    elif not width:
        r = height / float(h)
        dim = (int(w * r), height)

    elif not height:
        r = width / float(w)
        dim = (width, int(h * r))

    else:
        if w > h:
            img = resize(img, width=width)
        else:
            img = resize(img, height=height)
        h, w = img.shape
        dX = int(max(0, 32 - w) / 2.0)
        dY = int(max(0, 32 - h) / 2.0)
        method = cv2.BORDER_CONSTANT
        if padSame:
            method = cv2.BORDER_REPLICATE
        img = cv2.copyMakeBorder(img, top=dY, bottom=dY, left=dX, right=dX, borderType=method)
        dim = (width, height)

    return cv2.resize(img, dim, interpolation=inter)


def sort_contours(contours, method="left-to-right"):
    reverse = False
    i = 0
    possibilities = ["left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"]
    assert method in possibilities
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == possibilities[2] or method == possibilities[3]:
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    cnts, boundingBoxes = zip(*sorted(zip(contours, boundingBoxes), key=lambda x: (x[1][i], x[1][1]), reverse=reverse))
    return cnts, boundingBoxes


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def plot_model(history, epochs, validation=True, accuracy=True, loss=True, title="Training Analytics"):
    plt.style.use("ggplot")
    plt.figure()
    if loss:
        plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
        if validation:
            plt.plot(np.arange(0, epochs), history.history["val_loss"], label="validation_loss")
    if accuracy:
        plt.plot(np.arange(0, epochs), history.history["accuracy"], label="accuracy")
        if validation:
            plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="validation_accuracy")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Metric Values")
    plt.legend()
    return plt


def deskew_digit(img, applyThreshold=True):
    if applyThreshold:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    (h, w) = img.shape[:2]
    moments = cv2.moments(img)
    if abs(moments['mu02']) < 1e-2:
        return img
    skew = moments["mu11"] / moments["mu02"]
    M = np.float32([
        [1, skew, -0.5 * w * skew],
        [0, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def deskew_text(img, applyThreshold=True):
    if applyThreshold:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        thresh = img
    cords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(cords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def pointInside(r1, r2, leverage):
    """
    Checks whether the rectangle r2 is inside r1.
    :param r1: tuple of the form (x,y,w,h)
    :param r2: tuple of the form (x,y,w,h)
    :return: True if inside else False
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x1 - leverage <= x2 <= x1 + w1 + leverage and y1 - leverage <= y2 <= y1 + h1 + leverage:
        return True
    elif x1 - leverage <= x2 + w2 <= x1 + w1 + leverage and y1 - leverage <= y2 + h2 <= y1 + h1 + leverage:
        return True
    elif x1 - leverage <= x2 <= x1 + w1 + leverage and y1 - leverage <= y2 + h2 <= y1 + h1 + leverage:
        return True
    elif x1 - leverage <= x2 + w2 <= x1 + w1 + leverage and y1 - leverage <= y2 <= y1 + h1 + leverage:
        return True
    return False


def mergeRects(r1, r2):
    """
    merge 2 rectangles r1 and r2 where r1 has lower x co-ordinate.
    :param r1: first rectangle
    :param r2: second rectangle
    :return: merged rectangle
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    x3 = min(x1, x2)
    y3 = min(y1, y2)
    w3 = max(x1 + w1, x2 + w2) - x3
    h3 = max(y1 + h1, y2 + h2) - y3
    return x3, y3, w3, h3


def getExternalContours(img=None, cnts=None, leverage=3, applySort=False, minArea=None, display=False, original=False):
    col = (0, 255, 0)
    if img is not None and cnts is None:
        col = 255
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts, boundingBoxes = sort_contours(cnts)
    else:
        if not cnts:
            raise AssertionError("Please provide either the image or the contours.")
        if applySort:
            cnts, boundingBoxes = sort_contours(cnts)
        else:
            boundingBoxes = [cv2.boundingRect(cnt) for cnt in cnts]
    # Assuming the contours are sorted left to right and top ro bottom so the parent contour comes before its children.
    if original:
        return cnts, boundingBoxes
    contours = [cnts[0]]
    boundingBoxes1 = [boundingBoxes[0]]
    for i in range(1, len(boundingBoxes)):
        x, y, w, h = boundingBoxes[i]
        if minArea and w * h < minArea:
            continue
        if not pointInside(boundingBoxes1[-1], boundingBoxes[i], leverage):
            contours.append(cnts[i])
            boundingBoxes1.append(boundingBoxes[i])
        else:
            boundingBoxes1[-1] = mergeRects(boundingBoxes1[-1], boundingBoxes[i])
    if display:
        if img is None:
            print("No image given to display.")
        else:
            displayBoundingBoxes(img, boundingBoxes1, col)
            displayBoundingBoxes(img, boundingBoxes, col)
    return contours, boundingBoxes1


def displayBoundingBoxes(img, boundingBoxes, col=(0, 255, 0)):
    img1 = img.copy()
    for b in boundingBoxes:
        x, y, w, h = b
        cv2.rectangle(img1, (x, y), (x + w, y + h), col, 1)
    showImage(img1)


def centralize_digit(img, size):
    (eW, eH) = size
    if img.shape[1] > img.shape[0]:
        image = resize(img, width=eW, inter=cv2.INTER_AREA)
    else:
        image = resize(img, height=eH, inter=cv2.INTER_AREA)
    extent = np.zeros((eH, eW), dtype="uint8")
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image
    M = cv2.moments(extent)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)
    return extent


def deskew_digits(imgs, applyThreshold=False):
    return np.array([deskew_digit(img, applyThreshold) for img in imgs])


def centralize_digits(imgs, size):
    return np.array([centralize_digit(img, size) for img in imgs])
