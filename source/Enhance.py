import cv2 as cv
import numpy as np

from enhancer.motion_blur import estimateMotionBlur
from enhancer.noise_removal import removeNoise

# helper function for brighten()
def gamma_correction(image, gamma=0.5):
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in np.arange(256)]
    ).astype("uint8")

    return cv.LUT(image, table)


def brighten(image):
    gamma_img = gamma_correction(image, gamma=0.5)

    lab = cv.cvtColor(gamma_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_enh = clahe.apply(l)

    lab_enh = cv.merge((l_enh, a, b))

    return cv.cvtColor(lab_enh, cv.COLOR_LAB2BGR)


def restore_motion_blur(image):
    restored_image, _, _ = estimateMotionBlur(image)

    return restored_image


def derain(image, h=18, sharpen_amt=1.0):
    denoised = cv.fastNlMeansDenoisingColored(
        image,
        None,
        h=h,
        hColor=h,
        templateWindowSize=7,
        searchWindowSize=21
    )

    blurred = cv.GaussianBlur(
        denoised,
        (0, 0),
        sigmaX=3
    )

    sharpened = cv.addWeighted(
        denoised,
        1 + sharpen_amt,
        blurred,
        -sharpen_amt,
        0
    )

    return sharpened


def dehaze(image):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_eq = clahe.apply(l)

    lab_eq = cv.merge([l_eq, a, b])

    return cv.cvtColor(lab_eq, cv.COLOR_LAB2BGR)


def denoise(image):
    restoredImage, kernelSize, restore = removeNoise(image)

    return restoredImage

def enhance(image, degradations):
    if len(degradations) == 0:
        return image

    for degradation in degradations:

        match degradation:
            case "motion-blur":
                image = restore_motion_blur(image)
                print("motion-blur detected")

            case "noisy":
                image = denoise(image)
                print("noisy detected")

            case "low-light":
                image = brighten(image)
                print("low-light detected")

            case "haze":
                image = dehaze(image)
                print("haze detected")

            case "rain":
                image = derain(image)
                print("rain detected")

            case "snow":
                image = desnow(image)
                print("snow detected")

            case _:
                print(
                    f"Unknown degradation: {degradation}"
                )

    return image
