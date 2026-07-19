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

# Different countries and even states/cities implement different
# fonts and standards, for example in China all O-shaped characters
# are guaranteed to be a 0, all I-shaped characters are guaranteed
# to be 1.
#
# Meanwhile in Brazil (prior to September 2018), the format is LLL-NNNN
# where L is letter, N is number.
def _apply_country_corrections(text, country):
    if not text or not country:
        return text

    if country == "CN":
        corrections = {
            'O': '0',
            'I': '1'
        }
        return ''.join(corrections.get(c, c) for c in text)

    if country == "BR":
        if len(text) == 7:
            corrected = []
            for i, c in enumerate(text):
                if i < 3:
                    if c == '0':
                        corrected.append('O')
                    elif c == '1':
                        corrected.append('I')
                    elif c == '4':
                        corrected.append('A')
                    elif c == '5':
                        corrected.append('S')
                    elif c =='6':
                        corrected.append('G')
                    elif c == '8':
                        corrected.append('B')
                    else:
                        corrected.append(c)
                else:
                    if c == 'O':
                        corrected.append('0')
                    elif c == 'I':
                        corrected.append('1')
                    elif c == 'A':
                        corrected.append('4')
                    elif c == 'S':
                        corrected.append('5')
                    elif c == 'G':
                        corrected.append('6')
                    elif c == 'B':
                        corrected.append('8')
                    else:
                        corrected.append(c)
            return ''.join(corrected)
        else:
            return text

    return text

def _filter_detections(texts, scores, country=None):
    detections = []
    for text, score in zip(texts, scores):
        clean_text = ''.join(c for c in text if c.isascii() and c.isalnum())
        
        if clean_text:
            if country:
                clean_text = _apply_country_corrections(clean_text, country)
            detections.append((clean_text, score))

    return detections

def read_plate(image, country=None):
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
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])

    if not texts or not scores:
        return []

    detections = _filter_detections(texts, scores, country)

    if detections:
        best_text, best_score = max(detections, key=lambda x: x[1])
        print(f"License Plate: {best_text} | {best_score:.3f}\n")
        return [(best_text, best_score)]

    return []

def read_plate_pipeline(image, country=None):
    if image is None:
        return []

    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    result = ocr.predict(image)

    if not result or len(result) == 0 or result[0] is None:
        return []

    page = result[0]
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])

    if not texts or not scores:
        return []

    detections = _filter_detections(texts, scores, country)

    return [{'text': t, 'score': s} for t, s in detections]
