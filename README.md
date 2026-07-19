# ECE4512 Group 8 Final Project

Our project presents an adaptive image restoration pipeline for robust license plate recognition under adverse imaging conditions commonly encountered in traffic surveillance systems. While Automatic License Plate Recognition (ALPR) tools perform well on high-quality images, their performance degrades significantly in real-world scenarios where input images are captured under uncontrolled environmental conditions.

To address these limitations, we created a unified pipeline that combines degradation-aware image restoration with license plate detection and optical character recognition (OCR).

---

## Instructions

#### Prerequisites
- **Python 3.13.0** – we recommend using [pyenv](https://github.com/pyenv/pyenv) to manage your Python version:
  ```shell
  pyenv install 3.13.0
  pyenv local 3.13.0
  ```
- **Git** (to clone the repository)

#### Setup

1. **Clone the repository**
   ```shell
   git clone https://github.com/Ferrarr/ECE4512-Final-Project-Group-8
   cd ECE4512-Final-Project-Group-8
   ```

2. **Create and activate a virtual environment** (optional but recommended)
   ```shell
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```shell
   pip install -r requirements.txt
   ```

4. **Download the degradation‑classification model**  
   The pipeline requires the pre‑trained DACLIP model (~1.7 GB).  
   - Download [`daclip_ViT-B-32.pt`](https://huggingface.co/spaces/fffiloni/DA-CLIP/tree/main/pretrained_daclip_uir)  
   - Place it inside the `source/models/` folder

5. **Prepare your input images**  
   - Create the folder `assets/input_images/` (it is not provided by default):
     ```shell
     mkdir -p assets/input_images
     ```
   - Place all the images you want to process inside this folder.  
     Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`.

#### Run the pipeline

From the project root directory, execute:

**1. Automated pipeline (default)** – uses the CLIP-based classifier to detect degradations:
```shell
python source/main.py
```

**2. Demo mode** – processes all images inside `assets/demo/` instead of `assets/input_images/`:
```shell
python source/main.py demo
```

**3. Manual degradation override** – skips the classifier and forces specific restoration techniques. Useful for testing individual degradations:
```shell
python source/main.py --haze --rain
```
**4. Country-specific OCR corrections** – To improve reading accuracy for plates from specific countries (currently only supports Brazil and China), add the --BR or --CN flag. These will apply corrections for alphanumerics that look similar such as 0 and O, 1 and I, etc.
```shell
python source/main.py --CN --haze
```

You can combine any number of the available flags. The pipeline will apply all listed degradations to every image in the input directory.

**Available flags:**
- `--haze`
- `--rain`
- `--motion-blur`
- `--low-light`
- `--noisy`

> **Example:** `python source/main.py --haze --motion-blur --low-light`

#### Run the evaluation

To benchmark the pipeline's performance across baseline, automated, and manual experiments:

**1. Run the evaluation** – place desired benchmark images in `assets/test_set/` and run:
```shell
python source/evaluate.py
```

The evaluation runs:
- **Baseline** – no restoration applied
- **Automated** – DACLIP classifier + restoration
- **Manual** – ground-truth degradations from filename + restoration

**Test set naming convention:**
Images must follow the format: `degradation1_degradation2_<GROUND_TRUTH>.jpg`

Examples:
- `haze_AED-632.jpg`
- `motionblur_lowlight_ABC-1234.jpg`
- `rain_motionblur_haze_KVW-4909.jpg`

Available degradation tokens: `haze`, `rain`, `motionblur`, `lowlight`, `noisy`

**Output:**
The evaluation prints a summary table showing full-plate and character-level accuracy for each experiment.

```
======================================================================
EVALUATION SUMMARY
======================================================================

BASELINE (50 images):
  Full-plate accuracy: 0.420 (21/50)
  Character accuracy: 0.651

AUTOMATED (50 images):
  Full-plate accuracy: 0.680 (34/50)
  Character accuracy: 0.824

MANUAL (50 images):
  Full-plate accuracy: 0.920 (46/50)
  Character accuracy: 0.967
======================================================================
```

#### Output

- **OCR results** – recognised plate numbers and their confidence scores are printed directly to the terminal.
- **Cropped plate images** – each detected plate is saved as a separate `.jpg` file inside `assets/output_images/` with a filename like `<original_name>_plate_<id>.jpg`.

> **Note:** The `assets/output_images/` folder is created automatically if it does not exist, and its contents are cleared before every new run.

## Installation
- [daclip_ViT-B-32.pt](https://huggingface.co/spaces/fffiloni/DA-CLIP/tree/main/pretrained_daclip_uir)

## Credits
This project utilizes the following ...
- [daclip-uir](https://github.com/Algolzw/daclip-uir) for image degradation type recognition
