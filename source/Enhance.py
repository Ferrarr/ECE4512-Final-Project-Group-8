import cv2 as cv
import numpy as np

from enhancer.motion_blur import estimateMotionBlur
from enhancer.rain import remove_rain_fft

def gamma_correction(image, gamma=0.5):
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in np.arange(256)]
    ).astype("uint8")

    return cv.LUT(image, table)

def brighten(image):
    if len(image.shape) == 2:
        gamma_img = gamma_correction(image, gamma=0.5)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gamma_img)
    else:
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
    if len(image.shape) == 2:
        color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        restored = estimateMotionBlur(color)
        if restored is None:
            return image
        return cv.cvtColor(restored, cv.COLOR_BGR2GRAY)
    else:
        restored_image = estimateMotionBlur(image)
        if restored_image is None:
            return image
        return restored_image

def derain(image):
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image format")
    derained = remove_rain_fft(gray)
    return derained

def get_dark_channel(img, patch_size=15):
    min_channel = np.min(img, axis=2)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv.erode(min_channel, kernel)

    return dark_channel

def get_atmospheric_light(img, dark_channel, top_percent=0.001):
    h, w = dark_channel.shape
    num_pixels = h * w
    num_top = max(int(num_pixels * top_percent), 1)

    dark_flat = dark_channel.reshape(num_pixels)
    img_flat = img.reshape(num_pixels, 3)

    indices = np.argsort(dark_flat)[-num_top:]
    brightest = img_flat[indices]
    A = brightest[np.argmax(brightest.sum(axis=1))]

    return A.astype(np.float64)

def get_transmission(img, A, omega=0.95, patch_size=15):
    norm_img = img.astype(np.float64) / A

    dark_channel = get_dark_channel(norm_img, patch_size)
    transmission = 1 - omega * dark_channel

    return transmission

def guided_filter(guide, src, radius=40, eps=1e-3):
    guide = guide.astype(np.float64) / 255.0
    src = src.astype(np.float64)

    mean_guide = cv.boxFilter(guide, cv.CV_64F, (radius, radius))
    mean_src = cv.boxFilter(src, cv.CV_64F, (radius, radius))
    mean_gs = cv.boxFilter(guide * src, cv.CV_64F, (radius, radius))
    cov_gs = mean_gs - mean_guide * mean_src

    mean_gg = cv.boxFilter(guide * guide, cv.CV_64F, (radius, radius))
    var_g = mean_gg - mean_guide * mean_guide

    a = cov_gs / (var_g + eps)
    b = mean_src - a * mean_guide

    mean_a = cv.boxFilter(a, cv.CV_64F, (radius, radius))
    mean_b = cv.boxFilter(b, cv.CV_64F, (radius, radius))

    return mean_a * guide + mean_b

def recover_radiance(img, A, transmission, t0=0.1):
    t = np.clip(transmission, t0, 1.0)
    t = t[:, :, np.newaxis]

    J = (img.astype(np.float64) - A) / t + A

    return np.clip(J, 0, 255).astype(np.uint8)

def dehaze_dcp(img_rgb, patch_size=15, omega=0.95, t0=0.1, refine=True):
    dark_channel = get_dark_channel(img_rgb, patch_size)
    A = get_atmospheric_light(img_rgb, dark_channel)
    transmission = get_transmission(img_rgb, A, omega, patch_size)

    if refine:
        gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        transmission = guided_filter(gray, transmission)

    return recover_radiance(img_rgb, A, transmission, t0)

def gamma_correction_dcp(img_rgb, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]
    ).astype(np.uint8)

    return cv.LUT(img_rgb, table)

def dehaze(image):
    if len(image.shape) == 2:
        return image
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    dehazed_rgb = dehaze_dcp(img_rgb, patch_size=15, omega=0.95, t0=0.1)

    enhanced_rgb = gamma_correction_dcp(dehazed_rgb, gamma=1.5)

    blurred = cv.GaussianBlur(enhanced_rgb, (0, 0), sigmaX=3)
    sharpened = cv.addWeighted(enhanced_rgb, 1.7, blurred, -0.7, 0)

    return cv.cvtColor(sharpened, cv.COLOR_RGB2BGR)

def denoise(image):
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image format")

    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    denoised = clahe.apply(blurred)
    return denoised

def enhance(image, degradations):
    if len(degradations) == 0:
        return image

    priority = {
        "low-light": 1,
        "noisy": 2,
        "haze": 3,
        "rain": 4,
        "motion-blur": 5,
    }

    sorted_degradations = sorted(degradations, key=lambda d: priority.get(d, 99))

    for degradation in sorted_degradations:
        match degradation:
            case "motion-blur":
                image = restore_motion_blur(image)

            case "noisy":
                image = denoise(image)

            case "low-light":
                image = brighten(image)

            case "haze":
                image = dehaze(image)

            case "rain":
                image = derain(image)

            case _:
                print(
                    f"Unknown degradation: {degradation}"
                )

    return image
