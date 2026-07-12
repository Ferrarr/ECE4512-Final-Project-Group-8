from Extractor import extract
from Classifier import classify
# from Enhance import enhance
import Enhance as en
import cv2 as cv
import os
from datetime import datetime
from OCR import read_plate

def main():
    # image = cv.imread("0002a5b67e5f0909_jpg.rf.c8f81ef986e3e99af6f349c200080453.jpg")
    image = cv.imread("0190625-89_89-209&291_508&367-509&381_213&382_208&295_504&294-0_0_25_27_33_32_25-59-19.jpg")
    # image = cv.imread("blur10.png")
    plates, confidences = extract(image)

    for plate in plates:
        cv.imwrite('Extracted.jpg', plate)

        en.deblur(plate)
        read_plate(plate)

        # cv.imwrite('Output.jpg', plate)
        # read_plate('Output.jpg')

# def main():
#     directory = "../assets/blurred"
#     for image in os.listdir(directory):
#         image = os.path.join(directory, image)
#
#         if not os.path.isfile(image):
#             print("Not a File")
#             continue
#
#         plates, confidences = extract(image)
#
#         for plate in plates:
#             # degradations = classify(plate)
#             #
#             # enhanced_plate = enhance(plate, degradations)
#
#             # save image to ../output/ folder.
#             # os.makedirs("../output", exist_ok=True)
#             filename = "../output/enhanced_image_" + str(datetime.now().strftime("%H-%M-%S-%f")) + ".jpg"
#             # cv.imwrite(filename, plate)
#
#             read_plates(plate)


main()
