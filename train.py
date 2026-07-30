import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor

def train():
    # 1. Select Device (Force NVIDIA RTX 2050 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Load Pre-trained ViT Feature Extractor / Processor
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)

    # 3. Define Image Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # 4. Load Dataset from Folders
    train_dataset = datasets.ImageFolder(root='./dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    class_names = train_dataset.classes
    print(f"Detected Classes ({len(class_names)}): {class_names}")

    # 5. Load Pre-Trained Model with custom output class count
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # 6. Set Up Optimizer & Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # 7. Training Loop
    epochs = 5  # Fine-tuning only needs 3-5 epochs
    model.train()
    
    print("\nStarting Training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # 8. Save Fine-Tuned Model and Processor locally
    save_path = "./my_finetuned_vit"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"\nModel training complete! Saved to '{save_path}'")

if __name__ == "__main__":
    train()