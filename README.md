# ECE4512 Group 8 Final Project

## Instructions

### Prerequisites
- **Python 3.13.0** – we recommend using [pyenv](https://github.com/pyenv/pyenv) to manage your Python version:
  ```shell
  pyenv install 3.13.0
  pyenv local 3.13.0
  ```
- **Git** (to clone the repository)

### Setup

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
   On Windows, you may need:
   ```shell
   pip install -r requirements-win.txt
   pip install torch==2.12.1+cu130 torchvision==0.27.1+cu130 --index-url https://download.pytorch.org/whl/cu130
   ```

4. **Download the pre‑trained models**

   The following models are required:

   | Model | Source | Size | Location |
   |-------|--------|------|----------|
   | **DA‑CLIP** (degradation classifier) | [DA‑CLIP on Hugging Face](https://huggingface.co/spaces/fffiloni/DA-CLIP/tree/main/pretrained_daclip_uir) | ~1.7 GB | `source/models/daclip_ViT-B-32.pt` |
   | **License Plate Detector** (YOLOv11) | [morsetechlab on Hugging Face](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/tree/main) | Included in repo | `source/models/license-plate-finetune-v1n.pt` |

   > **Important:** The license plate detector model is already included in the repository. Only the DA‑CLIP model needs to be downloaded manually.

5. **Prepare your test images**

   - **For evaluation:** The `assets/eval/` folder is provided and contains pre-organised test images for each degradation condition. No additional setup is required.
   - **For single-image processing (optional):** If you wish to run the full pipeline on your own images, create the folder `assets/input_images/` and place your images inside. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`.

---

### Run the evaluation (Recommended)

To benchmark the pipeline's performance across baseline, automated, and assisted experiments:

```shell
python source/evaluate.py
```

**Optional country-specific OCR corrections:**
```shell
python source/evaluate.py --CN    # China corrections (0↔O, 1↔I)
python source/evaluate.py --BR    # Brazil corrections (LLL-NNNN format)
```

The evaluation runs three experiments per image:
- **Baseline** – no restoration applied
- **Assisted** – ground‑truth degradations parsed from the filename + restoration
- **Automated** – DACLIP classifier + restoration

**Test set naming convention:**  
Images must follow the format: `<degradation1>_<degradation2>_<GROUND_TRUTH>.jpg`

Examples:
- `haze_AED-632.jpg`
- `motionblur_lowlight_ABC-1234.jpg`
- `rain_motionblur_haze_KVW-4909.jpg`

Available degradation tokens: `haze`, `rain`, `motionblur`, `lowlight`, `noisy`

**Output:**  
Results are saved to `assets/evaluation_output/`, including:
- Processed plate crops for each experiment
- Detailed per-image results in a `.txt` file
- Summary statistics showing full-plate and character-level accuracy

Example summary:
```
======================================================================
EVALUATION SUMMARY
======================================================================

BASELINE (44 images):
  Full-plate accuracy: 0.205 (9/44)
  Character accuracy: 0.381

ASSISTED (44 images):
  Full-plate accuracy: 0.273 (12/44)
  Character accuracy: 0.586

AUTOMATED (44 images):
  Full-plate accuracy: 0.318 (14/44)
  Character accuracy: 0.549

----------------------------------------------------------------------
CLASSIFIER EVALUATION (Automated mode only)
  Average Label Accuracy: 0.333
  Average Jaccard Index (IoU): 0.333
  (based on 44 images)

----------------------------------------------------------------------
OVERALL (All modes):
  Full-plate accuracy: 0.265 (35/132)
  Character accuracy: 0.505
======================================================================
```

---

### Run the full pipeline (Optional)

To process images through the complete restoration and recognition pipeline:

```shell
python source/main.py
```

#### 1. Automated pipeline – uses the CLIP‑based classifier to detect degradations:
```shell
python source/main.py
```

#### 2. Assisted degradation override – forces specific restoration techniques (useful for testing):
```shell
python source/main.py --haze --rain
```

#### 3. Country‑specific OCR corrections:
```shell
python source/main.py --CN --haze
```

**Available flags:**
- `--haze`
- `--rain`
- `--motion-blur`
- `--low-light`
- `--noisy`
- `--CN` (China corrections)
- `--BR` (Brazil corrections)

**Output:**
- OCR results printed to terminal
- Cropped plate images saved to `assets/output_images/`

---

## Acknowledgements & Credits

This project builds upon several open‑source libraries and pre‑trained models. We gratefully acknowledge the contributions of:

- **[DA‑CLIP](https://github.com/Algolzw/daclip-uir)** – for degradation classification.  
  *Reference: Luo, Z., et al. "DA‑CLIP: Towards Degradation‑Aware CLIP for Universal Image Restoration."*

- **[YOLOv11 License Plate Detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)** – pre-trained model by morsetechlab for robust plate localisation.

- **[Rodosol ALPR Dataset](https://github.com/raysonlaroca/rodosol-alpr-dataset)** – Brazilian license plate dataset used for evaluation of the `--BR` country corrections.

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** – for optical character recognition.

- **[OpenCLIP](https://github.com/mlfoundations/open_clip)** – for the CLIP model implementation.
