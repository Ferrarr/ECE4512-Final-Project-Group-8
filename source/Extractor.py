from ultralytics import YOLO

try: 
    model = YOLO("./model/license-plate-finetune-v1n.pt")
    print('Model Loaded Succesfully')
except Exception as e: 
    print(f'Failed to load model: {e}')
    model = None

def extract(image_path):
    results = model.predict(image_path)

    plates = []
    confidences = []

    for result in results:
        image = result.orig_img

        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            plate = image[y1:y2, x1:x2]
            plates.append(plate)
            confidences.append(float(box.conf[0]))

    return plates, confidences

