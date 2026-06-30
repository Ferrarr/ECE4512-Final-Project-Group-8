from Extractor import extract
from Classifier import classify
from Enhance import enhance
import cv2 as cv
import os
from datetime import datetime

def main():
    image = "0002a5b67e5f0909_jpg.rf.c8f81ef986e3e99af6f349c200080453.jpg"
    plates, confidences = extract(image)

    for plate in plates:
        degradations = classify(plate)

        enhanced_plate = enhance(plate, degradations)

        # save image to output/ folder.
        os.makedirs("output", exist_ok=True)
        filename = "output/enhanced_image_" + str(datetime.now().strftime("%H-%M-%S-%f")) + ".jpg"
        cv.imwrite(filename, enhanced_plate)

main()
