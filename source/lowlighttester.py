import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Extractor import extract
from OCR import read_plate

img = cv2.imread("../assets/low light/lowlight5_img.jpg")
if img is None:
    print("Image not found!")
    exit()

def gamma_correction(image, gamma=0.5):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

gamma_img = gamma_correction(img, gamma=0.5)

lab = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l_enh = clahe.apply(l)
lab_enh = cv2.merge((l_enh, a, b))
clahe_img = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

cv2.imwrite("lowlight_output.png", clahe_img)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB))
plt.title("Gamma")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.savefig("lowlight_figure.png", dpi=250, bbox_inches='tight')

ext_img, _ = extract(clahe_img)
cv2.imwrite("lowlight_extract.png", ext_img[0])

ocr_plate = read_plate(ext_img[0])
print(ocr_plate)


