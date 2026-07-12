import cv2
import numpy as np

def gamma_correction(image, gamma=0.5):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def low_light(image):
    gamma_img = gamma_correction(img, gamma=0.5)

    lab = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge((l_enh, a, b))
    clahe_img = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    return clahe_img



