from Extractor import extract

# only import classify since that's the only function we need from Classifier in main
from Classifier import classify
import Enhance as en
import cv2 as cv


def main():
    image = "0002a5b67e5f0909_jpg.rf.c8f81ef986e3e99af6f349c200080453.jpg"
    plates, confidences = extract(image)

    cv.imshow("First plate found", plates[0])
    cv.waitKey(0)
    cv.imshow("First plate found", plates[1])
    cv.waitKey(0)
    cv.destroyAllWindows()

    return None


main()
