# ECE4512 Group 8 Final Project

Our project presents an adaptive image restoration pipeline for robust license plate recognition under adverse imaging conditions commonly encountered in traffic surveillance systems. While Automatic License Plate Recognition (ALPR) tools perform well on high-quality images, their performance degrades significantly in real-world scenarios where input images are captured under uncontrolled environmental conditions. 

To address these limitations, we created a unified pipeline that combines degradation-aware image restoration with license plate detection and optical character recognition (OCR). <continue this README>

To run the model, clone the repository and create a virtual environment with Python 3.10.0, then install the dependencies in requirements.txt
## Instructions

```shell
$ cd <folder>
$ git clone https://github.com/Ferrarr/ECE4512-Final-Project-Group-8
$ cd ECE4512-Final-Project-Group-8
$ mkdir .venv && python -m venv .venv
$ pip install -r requirements.txt
```

Then cd into source and run main.py
```shell
$ cd source && python3 main.py
```

## Installation
- [daclip_ViT-B-32.pt](https://huggingface.co/spaces/fffiloni/DA-CLIP/tree/main/pretrained_daclip_uir)

## Credits
This project utilizes the following ...
- [daclip-uir](https://github.com/Algolzw/daclip-uir) for image degradation type recognition
