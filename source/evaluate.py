import sys
from pathlib import Path
import cv2 as cv

from Classifier import classify
from Extractor import extract
from Enhance import enhance
from OCR import read_plate

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / "assets" / "test_set"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "evaluation_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEG_MAPPING = {
    "motionblur": "motion-blur",
    "lowlight": "low-light",
    "haze": "haze",
    "rain": "rain",
    "noisy": "noisy"
}

results = []


def get_test_images():
    if not TEST_DIR.exists():
        return []
    return [
        path for path in TEST_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def parse_filename(filename):
    stem = filename.stem
    parts = stem.split('_')

    if len(parts) < 2:
        return None, []

    ground_truth = parts[-1]
    deg_tokens = parts[:-1]

    degradations = []
    for token in deg_tokens:
        if token in DEG_MAPPING:
            degradations.append(DEG_MAPPING[token])

    return ground_truth, degradations


def run_pipeline(image_path, mode, forced_degs=None):
    image = cv.imread(str(image_path))
    if image is None:
        return None, None

    if mode == "baseline":
        enhanced_image = image
    elif mode == "manual" and forced_degs is not None:
        enhanced_image = enhance(image, forced_degs)
    else:
        degs = classify(image_path)
        enhanced_image = enhance(image, degs)

    plates = extract(enhanced_image)
    if not plates:
        detections = read_plate(enhanced_image)
    else:
        detections = read_plate(plates[0])

    if detections and len(detections) > 0:
        return detections[0][0], enhanced_image
    return None, enhanced_image


def compare_plate(pred, gt):
    if not pred or not gt:
        return False, 0.0

    pred_clean = ''.join(c for c in pred if c.isalnum()).upper()
    gt_clean = ''.join(c for c in gt if c.isalnum()).upper()

    if len(gt_clean) == 0:
        return False, 0.0

    matches = sum(1 for p, g in zip(pred_clean, gt_clean) if p == g)
    char_acc = matches / len(gt_clean)
    full_match = (pred_clean == gt_clean)

    return full_match, char_acc


def run_experiment(mode, images):
    print(f"\n--- Running {mode.upper()} ---")

    for img_path in images:
        gt, degs = parse_filename(img_path)

        if mode == "baseline":
            pred, _ = run_pipeline(img_path, "baseline")
        elif mode == "manual":
            pred, _ = run_pipeline(img_path, "manual", degs)
        else:
            pred, _ = run_pipeline(img_path, "automated")

        if gt and pred:
            full_match, char_acc = compare_plate(pred, gt)
            results.append({
                'filename': img_path.name,
                'mode': mode,
                'ground_truth': gt,
                'predicted': pred,
                'full_match': full_match,
                'char_acc': char_acc
            })
            status = "✓" if full_match else "✗"
            print(f"  {status} {img_path.name}: GT={gt} | Pred={pred} | Char={char_acc:.3f}")
        elif gt:
            results.append({
                'filename': img_path.name,
                'mode': mode,
                'ground_truth': gt,
                'predicted': None,
                'full_match': False,
                'char_acc': 0.0
            })
            print(f"  ✗ {img_path.name}: GT={gt} | Pred=(none)")


def print_summary():
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    modes = {"baseline": [], "manual": [], "automated": []}

    for r in results:
        if r['mode'] in modes:
            modes[r['mode']].append(r)

    for mode, entries in modes.items():
        if not entries:
            continue

        total = len(entries)
        full_matches = sum(1 for e in entries if e['full_match'])
        char_accs = [e['char_acc'] for e in entries]

        print(f"\n{mode.upper()} ({total} images):")
        print(f"  Full-plate accuracy: {full_matches / total:.3f} ({full_matches}/{total})")
        print(f"  Character accuracy: {sum(char_accs) / len(char_accs):.3f}")

    print("\n" + "-" * 70)
    print("OVERALL (All modes):")
    total = len(results)
    full_matches = sum(1 for e in results if e['full_match'])
    char_accs = [e['char_acc'] for e in results]
    print(f"  Full-plate accuracy: {full_matches / total:.3f} ({full_matches}/{total})")
    print(f"  Character accuracy: {sum(char_accs) / len(char_accs):.3f}")
    print("=" * 70)


def main():
    images = get_test_images()

    if not images:
        print(f"No test images found in {TEST_DIR}")
        return

    print(f"Found {len(images)} test images")

    # Experiment 1: Baseline (no restoration)
    run_experiment("baseline", images)

    # Experiment 2: Automated (DACLIP classifier)
    run_experiment("automated", images)

    # Experiment 3: Manual (ground truth from filename)
    run_experiment("manual", images)

    print_summary()


if __name__ == "__main__":
    main()
