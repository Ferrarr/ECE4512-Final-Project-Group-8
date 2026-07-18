import numpy as np 
import cv2 as cv 
from OCR import read_plate_pipeline

def blurAngleCepstrum(image): 
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    rows, cols = image.shape[:2]
    hanning = np.outer(np.hanning(rows), np.hanning(cols))
    img_w = image * hanning 

    Fourier = np.fft.fft2(img_w)
    Fshifted = np.fft.fftshift(Fourier)
    magni = np.abs(Fshifted)
    magni[magni == 0] = 1e-12
    logMag = np.log(magni)
    cepstrum = np.fft.ifft2(logMag).real
    cepstrumShift = np.fft.fftshift(cepstrum)

    cy, cx = np.array(cepstrumShift.shape) // 2
    y, x = np.ogrid[:rows, :cols]
    radius = 4
    mask = ((x-cx)**2 + (y-cy)**2) > radius**2
    cepMasked = cepstrumShift * mask

    peakIdx = np.unravel_index(
        np.argmax(cepMasked),
        cepstrumShift.shape
    )

    py, px = peakIdx
    dx = px - cx
    dy = py - cy

    angle_rad = np.arctan2(dy, dx)
    blur_angle = np.rad2deg(angle_rad) % 180
    correctedAngle = (180 - blur_angle) % 180 

    return correctedAngle


def motionBlurPSF(kernelSize, angle):
    kernelSize = int(kernelSize)
    psf = np.zeros((kernelSize, kernelSize), dtype=np.float32)
    center = kernelSize // 2
    psf[center, :] = 1 
    M = cv.getRotationMatrix2D((center, center), angle, 1.0)
    psf = cv.warpAffine(psf, M, (kernelSize, kernelSize))
    return psf / np.sum(psf)


def tenengrad(image): 
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    gx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

    return np.mean(gx**2 + gy**2)


def edgeTaper(image, psf, taperSize=None ): 
    # reduce Ringing (Softening the image's Edge)
    image = image.astype(np.float32) 
    height, width = image.shape[:2]
    ph, pw = psf.shape 

    if taperSize is None: 
        taperSize = max(ph,pw) *2
    
    padded = cv.copyMakeBorder(
        image, taperSize, taperSize, taperSize, taperSize,
        borderType=cv.BORDER_REFLECT
    )

    sigma = taperSize / 3.0 
    blurredVer = cv.GaussianBlur(padded, (0,0), sigmaX=sigma)

    mask = np.zeros_like(padded)
    mask[taperSize:-taperSize, taperSize:-taperSize] = 1
    mask = cv.GaussianBlur(mask, (0,0), sigmaX=sigma)

    tapered = padded * mask + blurredVer * (1 - mask)

    return tapered, taperSize


def wienerDeconvolution(image, psf, K=0.01): 

    if len(image.shape) == 3:
        channels = []

        for i in range(image.shape[2]):
            channel = wienerDeconvolution(
                image[:, :, i],
                psf,
                K
            )
            channels.append(channel)

        return np.stack(channels, axis=2)

    height, width = image.shape[:2]
    ph, pw = psf.shape

    psfPadded = np.zeros(
        (height, width),
        dtype=np.float32
    )

    cy, cx = height // 2, width // 2

    starty = cy - (ph // 2)
    startx = cx - (pw // 2) 

    psfPadded[
        starty:starty+ph,
        startx:startx+pw
    ] = psf 

    psfShifted = np.fft.ifftshift(psfPadded)

    H = np.fft.fft2(psfShifted)
    G = np.fft.fft2(image)

    HConj = np.conj(H)
    HAbs2 = np.abs(H) ** 2

    F_hat = (HConj / (HAbs2 + K)) * G 

    restored = np.fft.ifft2(F_hat).real 

    restored = np.clip(
        restored,
        0,
        255
    ).astype(np.uint8)

    return restored 


def restoreWithTapering(image, psf, K=0.01): 

    tapered_img, taperSize = edgeTaper(image, psf)

    restoredPadded = wienerDeconvolution(
        tapered_img,
        psf,
        K=K
    )

    height, width = image.shape[:2]

    if len(image.shape) == 3:
        restoredCropped = restoredPadded[
            taperSize:taperSize + height,
            taperSize:taperSize + width,
            :
        ]

        restoredFinal = np.zeros_like(restoredCropped)

        for i in range(restoredCropped.shape[2]):
            restoredFinal[:, :, i] = cv.normalize(
                restoredCropped[:, :, i],
                None,
                0,
                255,
                cv.NORM_MINMAX
            )

    else:
        restoredCropped = restoredPadded[
            taperSize:taperSize + height,
            taperSize:taperSize + width
        ]

        restoredFinal = cv.normalize(
            restoredCropped,
            None,
            0,
            255,
            cv.NORM_MINMAX
        )

    restoredFinal = restoredFinal.astype(np.uint8)

    return restoredFinal


def estimateLengthbySearch(image, angle, length_range = range(3,41), K=0.01): 

    allResult = []

    for L in length_range: 
        psf = motionBlurPSF(L, angle)
        restored = restoreWithTapering(
            image,
            psf,
            K=K
        )
        score = tenengrad(restored)
        allResult.append({
            'length' : L, 
            'score' : score, 
            'image' : restored
        })
    allResult.sort(key=lambda x: x['score'], reverse=True)
    topFive = allResult[:10]
        
    return topFive


def estimateMotionBlur(image):

    est_angle = blurAngleCepstrum(image)
    topFive = estimateLengthbySearch(image, est_angle)
    
    validOCR = []
    for rank, res in enumerate(topFive): 
        restoredImage = res['image']
        detections = read_plate_pipeline(restoredImage)
        if not detections: 
            continue 
        combinedText = ' '.join(d['text'] for d in detections)
        averageScore = sum(d['score'] for d in detections) / len(detections)
        if averageScore >= 0.75 : 
            validOCR.append({
                'image' : restoredImage, 
                'text' : combinedText, 
                'score' : averageScore, 
                'length' : res['length'], 
                'rank_tenegrad' : rank + 1
            })
            if averageScore >= 0.95: 
                break

    if not validOCR :
        return None        
    bestResult = max(validOCR, key=lambda x : x['score'])

    bestRestored = bestResult['image']
    plateResult = bestResult['text']
    bestScore = bestResult['score']
    bestLength = bestResult['length']
    return bestRestored

   
