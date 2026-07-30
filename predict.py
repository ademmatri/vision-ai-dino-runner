import os
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# 1. Load your class folder names from the train directory
train_dir = r"D:\Projects\vision_ai_project\dataset\train"
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# 2. Load saved model & processor
model_path = "./my_finetuned_vit"
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 3. Load test image
image_path = r"D:\Projects\vision_ai_project\tested_images\f.png"
image = Image.open(image_path).convert("RGB")

# 4. Run prediction
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()

# 5. Get actual class folder name
predicted_folder_name = class_names[predicted_class_id]

print(f"Predicted Class Index: {predicted_class_id}")
print(f"Predicted Folder Name: {predicted_folder_name}")