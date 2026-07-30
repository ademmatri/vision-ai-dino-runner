import cv2
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# 1. Load fine-tuned model and processor
model_path = "./my_finetuned_vit"
print("Loading model onto GPU/CPU...")
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Explicitly map class IDs to gesture names
# Swap 'Open' and 'Closed' order if 0 happens to be Open for your dataset
custom_labels = {
    2: "Hand Closed",
    3: "Hand Open"
}

# 2. Open webcam
cap = cv2.VideoCapture(0)

# Set resolution explicitly to help fix camera initialization issues
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n--- Live Camera Feed Active! ---")
print("Press 'q' in the window to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Warning: Skipping empty camera frame...")
        continue

    # Convert OpenCV BGR format to RGB for PIL/Transformers
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

    # Retrieve gesture label name (Fallback to custom_labels first)
    label_name = custom_labels.get(
        predicted_class_id, 
        model.config.id2label.get(predicted_class_id, f"Class {predicted_class_id}")
    )
    
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