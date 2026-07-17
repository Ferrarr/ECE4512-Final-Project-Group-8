import torch
from PIL import Image
import open_clip

checkpoint = "source/models/daclip_ViT-B-32.pt"

model, preprocess = open_clip.create_model_from_pretrained(
    "daclip_ViT-B-32",
    pretrained=checkpoint
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

def classify(image_path):
    image = preprocess(
        Image.open(image_path)
    ).unsqueeze(0)


    degradations = [
        "motion-blur",
        "haze",
        "low-light",
        "noisy",
        "rain",
    ]

    text = tokenizer(degradations)

    with torch.no_grad():
        text_features = model.encode_text(text)

        _, degra_features = model.encode_image(
            image,
            control=True
        )

        degra_features /= degra_features.norm(
            dim=-1,
            keepdim=True
        )

        text_features /= text_features.norm(
            dim=-1,
            keepdim=True
        )

        probs = (
            100.0 * degra_features @ text_features.T
        ).softmax(dim=-1)

    threshold = 0.45

    results = []

    for i, p in enumerate(probs[0]):
        confidence = p.item()

        if confidence > threshold:
            results.append(
                (
                    degradations[i],
                    confidence
                )
            )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [item[0] for item in results]
