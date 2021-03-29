from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import imutils
from PreProcessing import Preprocessor
from DigitAugment import DigitAugmenter

args = imutils.get_args(single_image=True)
img = cv2.imread(args["image"])
if img is None:
    raise FileNotFoundError("Please a valid path to the image.")

debug = False
method = 1

preprocessor = Preprocessor(640, 480, debug, method)
img, gray, canny, binary_img = preprocessor.preprocess(img)
img_ = img.copy()

augmenter = DigitAugmenter(transforms={"rotation": [-15, 15], "horizontal_shift": [-0.15, 0.15], "vertical_shift": [-0.15, 0.15]}, rescale=True)

minArea = 350
cnts, boundingBoxes = imutils.getExternalContours(img=binary_img, applySort=True, minArea=minArea, leverage=2, original=False)
number = 0
leverage = 10

models = [load_model(f"models\\digitnet_mnist_augment_decay_{i}.h5") for i in range(1, 6)]

for i, c in enumerate(cnts):
    x, y, w, h = boundingBoxes[i]
    aspect = w * 1.0 / h
    if w * h < minArea or aspect > 2.5 or aspect < 0.05:
        continue
    if method == 0:
        extract = gray[max(0, y - leverage):y + h + leverage, max(0, x - leverage):x + w + leverage]
        extract = cv2.threshold(extract, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        extract = binary_img[max(0, y - leverage):y + h + leverage, max(0, x - leverage):x + w + leverage]
    extracts = augmenter.generate_batch(extract, 16, process=True)
    pred = models[0].predict(extracts)
    for j in range(1, 5):
        pred += models[j].predict(extracts)
    pred = pred.sum(axis=0)
    pred = np.argmax(pred)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, str(pred), (x + w // 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)

imutils.showImage(img_, img, "Final Output", together=True)
save_name = args['image'].split("\\")[-1]
cv2.imwrite(f"outputs\\result_{save_name}", np.hstack((img_, img)))
