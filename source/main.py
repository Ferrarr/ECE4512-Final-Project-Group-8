import sys
from pathlib import Path
import cv2 as cv

from Classifier import classify
from Extractor import extract
from Enhance import enhance
from OCR import read_plate

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "assets" / "input_images"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "output_images"
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clear_output_directory(directory):
    if directory.exists():
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()

def get_images(directory):
    if not directory.exists():
        return []

    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

def process_image(image_path, forced_degradations=None, country=None):
    if forced_degradations is None:
        image_degradations = classify(image_path)
    else:
        image_degradations = forced_degradations

    image = cv.imread(str(image_path))
    if image is None:
        print(f"Failed to read {image_path}")
        return

    enhanced_image = enhance(image, image_degradations)

    plates = extract(enhanced_image)
    if not plates:
        filename = OUTPUT_DIR / (f"{image_path.stem}_plate_0.jpg")
        cv.imwrite(str(filename), enhanced_image)
        read_plate(enhanced_image, country=country)
    else:
        for plate_id, plate in enumerate(plates):
            filename = OUTPUT_DIR / (f"{image_path.stem}_plate_{plate_id}.jpg")
            cv.imwrite(str(filename), plate)
            read_plate(plate, country=country)

def main():
    clear_output_directory(OUTPUT_DIR)

    args = sys.argv[1:]
    is_demo = "demo" in args
    forced_degradations = None
    valid_flags = {"--haze", "--rain", "--motion-blur", "--low-light", "--noisy"}
    country = None

    for arg in args:
        if arg == "--CN":
            country = "CN"
        elif arg == "--BR":
            country = "BR"

    if not is_demo:
        degs = []
        for arg in args:
            if arg in valid_flags:
                degs.append(arg[2:])
        if degs:
            forced_degradations = degs

    if is_demo:
        input_dir = PROJECT_ROOT / "assets" / "demo"
    else:
        input_dir = INPUT_DIR

    inputs = get_images(input_dir)

    if not inputs:
        print("No images found.")
        return

    for image_path in inputs:
        process_image(image_path, forced_degradations, country)

if __name__ == "__main__":
    main()
