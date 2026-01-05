from ultralytics import YOLO
from pathlib import Path

# Load model
model = YOLO("best.pt")

# Paths
image_dir = Path("datasets/rdd2022/test/images")
output_dir = Path("predictions")
output_dir.mkdir(parents=True, exist_ok=True)

images = sorted(image_dir.glob("*.jpg"))

print(f"Total images: {len(images)}")

for i, img in enumerate(images, 1):
    model.predict(
        source=str(img),
        save_txt=True,
        save_conf=True,
        project=".",
        name="predictions",
        exist_ok=True,
        verbose=False
    )

    if i % 200 == 0:
        print(f"Processed {i}/{len(images)}")

print("Inference complete.")
