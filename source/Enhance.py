# I put all of the long code inside of the folder 
# "Enhancer" and im just going to import it here
# im thinking that this way will make 
#  the code to be much more neat 
from Enhancer.estimateMotionBlur import estimateMotionBlur
from Enhancer.noiseRemoval import removeNoise
import matplotlib.pyplot as plt 
from Extractor import extract
from OCR import read_plate

# function to brighten image in case of low light
def brighten(image):
    return 1


import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def deblur(image, psf=None, snr=0.01):
    """
    Perform Wiener deconvolution to remove blur.

    Parameters:
        image : numpy.ndarray
            Input image (grayscale: 2D, RGB: 3D with last axis = colour channels).
            Expected to be in [0, 1] float range.
        psf : numpy.ndarray, optional
            Point spread function (blur kernel). If None, a Gaussian kernel is used.
        snr : float, optional
            Noise-to-signal power ratio (default 0.01). Higher values suppress noise
            but reduce deblurring strength.

    Returns:
        numpy.ndarray
            Deblurred image, same shape and dtype as input.
    """
    # Handle colour images channel‑wise
    if image.ndim == 3:
        channels = [deblur(image[..., c], psf, snr) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)

    # Default PSF: 15×15 Gaussian blur (sigma = 1.5)
    if psf is None:
        size = 15
        sigma = 1.5
        ax = np.linspace(-(size // 2), size // 2, size)
        x, y = np.meshgrid(ax, ax)
        psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        psf /= psf.sum()          # normalise

    # Pad the PSF to the image size (centre‑placed)
    h, w = image.shape
    ph, pw = psf.shape
    psf_padded = np.zeros_like(image, dtype=np.float64)
    y_start = (h - ph) // 2
    x_start = (w - pw) // 2
    psf_padded[y_start:y_start+ph, x_start:x_start+pw] = psf

    # Fourier transforms
    F = fft2(image)
    H = fft2(psf_padded)

    # Wiener filter: G = H* / (|H|² + K)
    H_conj = np.conj(H)
    H_abs_sq = np.abs(H) ** 2
    H_abs_sq = np.maximum(H_abs_sq, 1e-12)   # avoid division by zero
    K = snr
    G = H_conj / (H_abs_sq + K)

    # Apply filter and inverse transform
    deblurred = np.real(ifft2(G * F))

    # Clip to valid range (assumes input is in [0,1])
    deblurred = np.clip(deblurred, 0, 1)

    return deblurred
    

# this function name sucks, we can change it later
def motionBlurRestore(image):
    restoredImage, selectedLength, est_angle = estimateMotionBlur(image)
    # plt.imshow(restoredImage)
    # plt.show()
    plates, confidences = extract(restoredImage)
    try: 
        plate = plates[0]
    except: 
        print('No Plate detected!')
        return None 
    plt.imshow(plate)
    plt.show()
    plateNum = read_plate(plate)
    print(f'Selected Length: {selectedLength} | Estimate Angle: {est_angle}')
    return 1


def derain(image):
    return 1


def super_resolution(image):
    return 1


def depixelate(image):
    return 1


def unglare(image):
    return 1


def denoise(image):
    restoredImage, kernelSize, restore = removeNoise(image) 
    plate ,confidences = extract(restoredImage)
    try: 
        plate = plate[0]
    except: 
        print('No plate detected!')
        return None 
    plt.imshow(plate)
    plt.show()
    plateNum = read_plate(plate)
    print(f'Kernel Size: {kernelSize}x{kernelSize} | Restoration: {restore}')
    return 1

def enhance(image, degradations):
    match degradations: 
        case 'motionBlur': 
            motionBlurRestore(image)
        case 'noise': 
            denoise(image)
        case _: 
            print(f'Unknown Info: {image}, {degradations}')
            return None

    return 1
