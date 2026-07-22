from ultralytics import YOLO

try: 
    model = YOLO("source/models/license-plate-finetune-v1n.pt")
    print('Model Loaded Succesfully')
except Exception as e: 
    print(f'Failed to load model: {e}')
    model = None

def extract(image):
    results = model.predict(image, conf=0.5, verbose=False)  
    plates = []

    for result in results:
        img = result.orig_img  

        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(box_w * 0.15)
            pad_y = int(box_h * 0.15)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img.shape[1], x2 + pad_x)
            y2 = min(img.shape[0], y2 + pad_y)

            plate = img[y1:y2, x1:x2]
            plates.append(plate)

    return plates
