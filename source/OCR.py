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
        print("OCR: Input image is None.")
        return None

    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    elif len(image.shape) != 3 or image.shape[2] != 3:
        print(f"OCR: Unexpected image shape:{image.shape}")
        return None

    result = ocr.predict(image)

    if len(result) == 0:
        return []

    page = result[0]

    texts = page["rec_texts"]
    scores = page["rec_scores"]

    detections = []

    for text, score in zip(texts, scores):
        detections.append((text, score))

    if detections:
        best_text, best_score = max(detections, key=lambda x: x[1])
        print(f"License Plate: {best_text} | {best_score:.3f}")
        return [(best_text, best_score)]

    return detections
