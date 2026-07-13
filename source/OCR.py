import paddleocr

# -------------------------------------------------------------------------
# HOTFIX (DO NOT REMOVE): Disable MKLDNN to avoid the oneDNN PIR conversion error.
# -------------------------------------------------------------------------
_orig_init = paddleocr.PaddleOCR.__init__

def _patched_init(self, *args, **kwargs):
    kwargs["enable_mkldnn"] = False
    _orig_init(self, *args, **kwargs)

paddleocr.PaddleOCR.__init__ = _patched_init
# -------------------------------------------------------------------------

from paddleocr import PaddleOCR
import cv2 as cv

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="en"
)


def read_plate(image):
    if image is None:
        print("Input image is None.")
        return None

    # PaddleOCR expects either grayscale or BGR
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    elif len(image.shape) != 3 or image.shape[2] != 3:
        print(f"Unexpected image shape: {image.shape}")
        return None

    result = ocr.predict(image)

    if len(result) == 0:
        return []

    page = result[0]

    texts = page["rec_texts"]
    scores = page["rec_scores"]

    detections = []

    # for text, score in zip(texts, scores):
    #     print(f"{text} | {score:.3f}")
    #     detections.append((text, score))

    return detections
