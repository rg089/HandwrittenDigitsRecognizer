import cv2
import numpy as np
import imutils


class Preprocessor:
    def __init__(self, width, height, debug=False, method=1):
        self.width = width
        self.height = height
        self.debug = debug
        self.method = method

    def convert_and_threshold_line(self, img_):
        """
        Takes Image, resizes it, converts to grayscale, cleans with divide tricks and thresholds it with Otsu's Threshold.
        :param img_: Unprepared Image
        :return: resized image, grayscale, cleaned image with divide trick, thresholded image.
        """
        img_ = cv2.resize(img_, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        cleaned = cv2.GaussianBlur(gray, (51, 51), 0)
        cleaned = cv2.divide(gray, cleaned, scale=255)
        if self.debug:
            imutils.showImage(cleaned, "After Divide Trick")
        thresh = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        if self.debug:
            imutils.showImage(thresh, "After Otsu's Thresholding on Divided")
        return img_, gray, cleaned, thresh

    def process_img_without_lines(self, gray_inv, canny):
        """
        Takes image with no grids or lines, thresholds it according to three possible methods. Method 0 - Canny, Method 1 - mu + sigma/2 threshold, Method 2 - Blackhat.
        :param gray_inv: Inverted version of the grayscale image.
        :param canny: Image after applying canny
        :return: Final Thresholded Binary Image.
        """
        if self.method == 2:
            gray = 255 - gray_inv
            rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 13))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
            thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            return thresh

        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, (13, 13))
        if self.method == 0:
            return canny
        cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ans = np.array([])
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            roi = gray_inv[y:y + h, x:x + w].ravel()
            ans = np.concatenate((ans, roi))
        mean, std = np.mean(ans), np.std(ans)
        thresh_val = int(mean + std / 2)
        output = np.zeros_like(gray_inv)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            roi = gray_inv[y:y + h, x:x + w]
            output[y:y + h, x:x + w] = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)[1]
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, (3, 3))
        return output

    def getDetectedLinesCanny(self, cleaned):
        """
        Takes image (cleaned after divide trick) and detects lines using Canny, contours and then morphing open.
        :param cleaned: cleaned input image
        :return: image after applying canny and binary detected lines.
        """
        v = np.median(cleaned)
        sigma = 0.33
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
        canny = cv2.Canny(cleaned, lower_thresh, upper_thresh)
        cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if self.debug:
            imutils.showImage(canny, "Canny")
        kern_width = 20
        new = np.zeros_like(canny)
        cv2.drawContours(new, cnts, -1, 255, 2)
        new = cv2.erode(new, (6, 2), iterations=2)
        if self.debug:
            imutils.showImage(new, "Contours on Canny")
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_width, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kern_width * 4))
        detected_lines = cv2.morphologyEx(new, cv2.MORPH_OPEN, horizontal_kernel, iterations=2) + cv2.morphologyEx(new, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        if self.debug:
            imutils.showImage(detected_lines, "Detected on Canny Contours(R)")
        return canny, detected_lines

    def remove_lines_final(self, thresh, detected_lines):
        """
        Remove given lines from the given thresholded image, clean and repair.
        :param thresh: The thresholded image containing numbers and lines.
        :param detected_lines: Binary image containing only lines.
        :return: Thresholded Binary Image after removing lines, cleaning and repairing.
        """
        thresh[detected_lines > 0] = 0
        if self.debug:
            imutils.showImage(thresh, "Remove Lines")
        thresh = cv2.medianBlur(thresh, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        if self.debug:
            imutils.showImage(thresh, "Final Image after Removing Lines")
        return thresh

    def anyDetectedLines(self, img_):
        """
        Checks if there is anything white in a binary image.
        :param img_: Binary Image
        :return: True if there is any white pixels present, else False.
        """
        return (img_ > 0).sum() > 0

    def preprocess(self, img):
        """
        Gives the final preprocessed version of image.
        :param img: Input Image
        :return: img, grayscale, canny and the final binary_img
        """
        img, gray, cleaned, thresh = self.convert_and_threshold_line(img)
        canny, detected_lines = self.getDetectedLinesCanny(cleaned)
        if not self.anyDetectedLines(detected_lines):
            thresh_final = self.process_img_without_lines(255 - gray, canny)
        else:
            thresh_final = self.remove_lines_final(thresh, detected_lines)
        if self.debug:
            imutils.showImage(thresh_final, "Final Processed Image.")
        return img, gray, canny, thresh_final
