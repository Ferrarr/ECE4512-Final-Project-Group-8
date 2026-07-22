import sys
import argparse
from pathlib import Path
import cv2 as cv
import Levenshtein

from Classifier import classify
from Extractor import extract
from Enhance import enhance
from OCR import read_plate

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "assets" / "eval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "evaluation_output"

DEG_MAPPING = {
    "motionblur": "motion-blur",
    "lowlight": "low-light",
    "haze": "haze",
    "rain": "rain",
    "noisy": "noisy"
}

def get_image_files(directory):
    if not directory.exists():
        return []
    return [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

def get_subfolders(directory):
    if not directory.exists():
        return []
    return [path for path in directory.iterdir() if path.is_dir()]

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

def compare_plate(pred, gt):
    if not pred or not gt:
        return False, 0.0

    pred_clean = ''.join(c for c in pred if c.isalnum()).upper()
    gt_clean = ''.join(c for c in gt if c.isalnum()).upper()

    if len(gt_clean) == 0:
        return False, 0.0

    distance = Levenshtein.distance(pred_clean, gt_clean)
    char_acc = 1 - (distance / max(len(pred_clean), len(gt_clean), 1))
    full_match = (distance == 0)

    return full_match, char_acc

def save_plate_image(plate, output_dir, filename, suffix):
    output_path = output_dir / f"{filename}_{suffix}.jpg"
    cv.imwrite(str(output_path), plate)
    return output_path

def process_image_for_eval(image_path, output_dir, mode, forced_degs=None, country=None):
    image = cv.imread(str(image_path))
    if image is None:
        return None, None, None

    stem = image_path.stem

    if mode == "baseline":
        degs = []
    elif mode == "assisted" and forced_degs is not None:
        degs = forced_degs
    else:
        degs = classify(image_path)

    plates = extract(image)

    if not plates:
        if mode == "baseline":
            enhanced_full = image
        else:
            enhanced_full = enhance(image, degs)

        plates = extract(enhanced_full)

        if not plates:
            full_output_path = output_dir / f"{stem}_full_{mode}.jpg"
            cv.imwrite(str(full_output_path), enhanced_full)

            detections = read_plate(enhanced_full, country=country)
            if detections and len(detections) > 0:
                pred_text = detections[0][0]
            else:
                pred_text = None

            return pred_text, enhanced_full, stem

        plate = plates[0]
        enhanced_plate = plate
        save_plate_image(enhanced_plate, output_dir, stem, mode)

    else:
        plate = plates[0]

        if mode == "baseline":
            enhanced_plate = plate
        else:
            enhanced_plate = enhance(plate, degs)

        save_plate_image(enhanced_plate, output_dir, stem, mode)

    detections = read_plate(enhanced_plate, country=country)
    if detections and len(detections) > 0:
        return detections[0][0], enhanced_plate, stem

    return None, enhanced_plate, stem

def run_experiment(mode, images, output_dir, country=None, forced_degs=None):
    results = []

    mode_label = mode.upper()
    print(f"\n--- Running {mode_label} ---")

    class_label_accs = []
    class_jaccards = []

    for img_path in images:
        gt, degs_from_file = parse_filename(img_path)

        if mode == "assisted":
            forced = degs_from_file if degs_from_file else forced_degs
        else:
            forced = None

        pred, _, stem = process_image_for_eval(
            img_path,
            output_dir,
            mode,
            forced_degs=forced,
            country=country
        )

        if mode == "automated":
            pred_degs = classify(img_path) if gt is not None else []
            gt_degs_set = set(degs_from_file)
            pred_degs_set = set(pred_degs)
            intersection = gt_degs_set & pred_degs_set
            union = gt_degs_set | pred_degs_set
            label_acc = len(intersection) / len(gt_degs_set) if gt_degs_set else 0.0
            jaccard = len(intersection) / len(union) if union else 1.0
            class_label_accs.append(label_acc)
            class_jaccards.append(jaccard)

            print(f"  Classifier: GT={sorted(degs_from_file)} | Pred={sorted(pred_degs)} | LabelAcc={label_acc:.3f} | IoU={jaccard:.3f}")

        if gt and pred:
            full_match, char_acc = compare_plate(pred, gt)
            result = {
                'filename': img_path.name,
                'stem': stem,
                'mode': mode,
                'ground_truth': gt,
                'predicted': pred,
                'full_match': full_match,
                'char_acc': char_acc
            }
            results.append(result)

            status = "✓" if full_match else "✗"
            print(f"  {status} {img_path.name}: GT={gt} | Pred={pred} | Char={char_acc:.3f}")

        elif gt:
            result = {
                'filename': img_path.name,
                'stem': stem,
                'mode': mode,
                'ground_truth': gt,
                'predicted': None,
                'full_match': False,
                'char_acc': 0.0
            }
            results.append(result)
            print(f"  ✗ {img_path.name}: GT={gt} | Pred=(none)")

    if mode == "automated" and class_label_accs:
        results.append({
            'mode': 'classifier_summary',
            'avg_label_accuracy': sum(class_label_accs) / len(class_label_accs),
            'avg_jaccard': sum(class_jaccards) / len(class_jaccards),
            'num_images': len(class_label_accs)
        })

    return results

def write_results_file(output_path, folder_name, all_results, country):
    with open(output_path, 'w') as f:
        f.write(f"Evaluation Results: {folder_name}\n")
        f.write(f"Country corrections: {country if country else 'None'}\n")
        total_images = len([r for r in all_results if r['mode'] in ["baseline", "assisted", "automated"]])
        f.write(f"Total images: {total_images}\n")
        f.write("\n" + "=" * 70 + "\n\n")

        for mode in ["baseline", "assisted", "automated"]:
            mode_results = [r for r in all_results if r['mode'] == mode]
            if not mode_results:
                continue

            f.write(f"--- {mode.upper()} ---\n")
            for r in mode_results:
                status = "✓" if r['full_match'] else "✗"
                if r['predicted']:
                    f.write(f"  {status} {r['filename']}: GT={r['ground_truth']} | ")
                    f.write(f"Pred={r['predicted']} | Char={r['char_acc']:.3f}\n")
                else:
                    f.write(f"  {status} {r['filename']}: GT={r['ground_truth']} | Pred=(none)\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n")

        modes = {"baseline": [], "assisted": [], "automated": []}
        for r in all_results:
            if r['mode'] in modes:
                modes[r['mode']].append(r)

        for mode, entries in modes.items():
            if not entries:
                continue

            total = len(entries)
            full_matches = sum(1 for e in entries if e['full_match'])
            char_accs = [e['char_acc'] for e in entries]

            f.write(f"\n{mode.upper()} ({total} images):\n")
            f.write(f"  Full-plate accuracy: {full_matches / total:.3f} ({full_matches}/{total})\n")
            f.write(f"  Character accuracy: {sum(char_accs) / len(char_accs):.3f}\n")

        classifier_summary = [r for r in all_results if r.get('mode') == 'classifier_summary']
        if classifier_summary:
            cs = classifier_summary[0]
            f.write("\n" + "-" * 70 + "\n")
            f.write("CLASSIFIER EVALUATION (Automated mode only)\n")
            f.write(f"  Average Label Accuracy: {cs['avg_label_accuracy']:.3f}\n")
            f.write(f"  Average Jaccard Index (IoU): {cs['avg_jaccard']:.3f}\n")
            f.write(f"  (based on {cs['num_images']} images)\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("OVERALL (All modes):\n")
        total = len([r for r in all_results if r['mode'] in ["baseline", "assisted", "automated"]])
        if total > 0:
            full_matches = sum(1 for e in all_results if e.get('full_match', False))
            char_accs = [e['char_acc'] for e in all_results if 'char_acc' in e]
            f.write(f"  Full-plate accuracy: {full_matches / total:.3f} ({full_matches}/{total})\n")
            if char_accs:
                f.write(f"  Character accuracy: {sum(char_accs) / len(char_accs):.3f}\n")
        f.write("=" * 70 + "\n")

def evaluate_folder(folder_path, output_dir, country=None):
    folder_name = folder_path.name
    images = get_image_files(folder_path)

    if not images:
        print(f"No images found in {folder_path}")
        return

    folder_output_dir = output_dir / folder_name
    folder_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Evaluating folder: {folder_name} ({len(images)} images)")
    print(f"Output: {folder_output_dir}")
    print(f"{'='*70}")

    all_results = []

    # Experiment 1: Baseline (no restoration)
    baseline_results = run_experiment(
        "baseline",
        images,
        folder_output_dir,
        country
    )
    all_results.extend(baseline_results)

    # Experiment 2: Assisted (force degradation from filename)
    assisted_results = run_experiment(
        "assisted",
        images,
        folder_output_dir,
        country
    )
    all_results.extend(assisted_results)

    # Experiment 3: Automated (DACLIP classifier)
    automated_results = run_experiment(
        "automated",
        images,
        folder_output_dir,
        country
    )
    all_results.extend(automated_results)

    result_file = folder_output_dir / f"{folder_name}_result.txt"
    write_results_file(result_file, folder_name, all_results, country)

    print(f"\nResults saved to {result_file}")

def clear_output_directory(directory):
    if directory.exists():
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                for subitem in item.iterdir():
                    if subitem.is_file():
                        subitem.unlink()
                    elif subitem.is_dir():
                        subitem.rmdir()
                item.rmdir()
    else:
        directory.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Evaluate license plate recognition.")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to folder containing subfolders of test images (default: assets/eval)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to output folder for result files (default: assets/evaluation_output)")
    parser.add_argument("--CN", action="store_true", help="Use China country corrections")
    parser.add_argument("--BR", action="store_true", help="Use Brazil country corrections")

    args = parser.parse_args()

    country = None
    if args.CN:
        country = "CN"
    elif args.BR:
        country = "BR"

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = DEFAULT_INPUT_DIR

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    clear_output_directory(output_dir)

    subfolders = get_subfolders(input_dir)

    if not subfolders:
        print(f"No subfolders found in {input_dir}")
        return

    print(f"Found {len(subfolders)} folders to evaluate")
    if country:
        print(f"Country corrections: {country}")

    for folder in subfolders:
        evaluate_folder(folder, output_dir, country)

    print(f"\n{'='*70}")
    print("All evaluations complete!")
    print(f"Results saved to {output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
