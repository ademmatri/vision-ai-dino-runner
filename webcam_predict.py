import cv2
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# 1. Load your fine-tuned model and processor
model_path = "./my_finetuned_vit"
print("Loading model onto GPU...")
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Get class labels from model configuration
id2label = model.config.id2label if hasattr(model.config, 'id2label') else {}

# 2. Open webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n--- Live Camera Feed Active! ---")
print("Press 'q' in the window to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV uses BGR format, convert to RGB for PIL/Transformers
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # Preprocess image frame
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    # Run real-time inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        predicted_class_id = logits.argmax(-1).item()
        confidence = probs[predicted_class_id].item() * 100

    # Retrieve label name
    label_name = id2label.get(predicted_class_id, f"Class {predicted_class_id}")
    display_text = f"Guess: {label_name} ({confidence:.1f}%)"

    # Overlay prediction text on top of the webcam feed
    cv2.putText(frame, display_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display video window
    cv2.imshow("Real-Time Vision Classifier", frame)

    # Press 'q' key to quit webcam loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()