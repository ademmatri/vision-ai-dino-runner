import gradio as gr
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# Load model & processor
model_path = "ademmatri/vit-hand-gesture-dino"
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def classify_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Return probabilities for all classes
    labels = list(model.config.id2label.values()) if hasattr(model.config, 'id2label') else [f"Class {i}" for i in range(len(probs))]
    return {labels[i]: float(probs[i]) for i in range(len(probs))}

# Create interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Vision Transformer Classifier"
)

interface.launch()
