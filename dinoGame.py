import cv2
import torch
import random
import pygame
import threading
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# ==========================================
# 1. LOAD FINE-TUNED VISION TRANSFORMER
# ==========================================
model_path = "./my_finetuned_vit"
print("Loading ViT model onto device...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained(model_path).to(device)
processor = ViTImageProcessor.from_pretrained(model_path)
model.eval()

id2label = model.config.id2label if hasattr(model.config, 'id2label') else {}

# Shared variables between threads
latest_prediction = "Initializing..."
lock = threading.Lock()
running = True

# ==========================================
# 2. BACKGROUND THREAD FOR VIT INFERENCE
# ==========================================
def vit_inference_thread():
    global latest_prediction, running
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Prepare frame for ViT model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_id = outputs.logits.argmax(-1).item()

        raw_label = id2label.get(predicted_class_id, str(predicted_class_id))
        
        # Safely update prediction string shared with Pygame
        with lock:
            latest_prediction = str(raw_label).lower()

    cap.release()

# Start background thread
ai_thread = threading.Thread(target=vit_inference_thread, daemon=True)
ai_thread.start()

# ==========================================
# 3. INITIALIZE PYGAME ENGINE
# ==========================================
pygame.init()
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Time ViT Dino Runner")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
GROUND_Y = 320
font = pygame.font.Font(None, 36)

class Dinosaur:
    def __init__(self):
        self.x = 80
        self.y = GROUND_Y - 50
        self.width = 40
        self.height = 50
        self.vel_y = 0
        self.gravity = 0.7
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.vel_y = -14
            self.is_jumping = True

    def update(self):
        self.vel_y += self.gravity
        self.y += self.vel_y
        if self.y >= GROUND_Y - self.height:
            self.y = GROUND_Y - self.height
            self.is_jumping = False
            self.vel_y = 0

    def draw(self, surface):
        pygame.draw.rect(surface, GREEN, (self.x, self.y, self.width, self.height), border_radius=5)
        pygame.draw.circle(surface, WHITE, (self.x + 30, self.y + 12), 4)

class Cactus:
    def __init__(self):
        self.x = WIDTH + random.randint(0, 100)
        self.width = 25
        self.height = random.randint(40, 60)
        self.y = GROUND_Y - self.height

    def update(self):
        self.x -= 4

    def draw(self, surface):
        pygame.draw.rect(surface, RED, (self.x, self.y, self.width, self.height), border_radius=3)

dino = Dinosaur()
cactuses = [Cactus()]
score = 0
game_over = False

# ==========================================
# 4. MAIN GAME LOOP (SMOOTH 60 FPS)
# ==========================================
while running:
    clock.tick(60) # High game frame rate

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_over:
            dino = Dinosaur()
            cactuses = [Cactus()]
            score = 0
            game_over = False

    # Get latest prediction from background thread
    with lock:
        current_pred = latest_prediction

    # TRIGGER JUMP: Check label or class string
    # (Adjust "open" if your dataset label is named differently, e.g., "1" or "hand_open")
    if "fist_opened" in current_pred or "label_3" in current_pred:
        if not game_over:
            dino.jump()

    # Game physics update
    if not game_over:
        dino.update()
        score += 1
        for cactus in cactuses:
            cactus.update()
            if pygame.Rect(dino.x, dino.y, dino.width, dino.height).colliderect(
                pygame.Rect(cactus.x, cactus.y, cactus.width, cactus.height)
            ):
                game_over = True

        if cactuses[-1].x < WIDTH - 300 and random.random() < 0.5:
            cactuses.append(Cactus())
        cactuses = [c for c in cactuses if c.x > -50]

    # Render graphics
    screen.fill(WHITE)
    pygame.draw.line(screen, BLACK, (0, GROUND_Y), (WIDTH, GROUND_Y), 3)
    dino.draw(screen)
    for c in cactuses:
        c.draw(screen)

    screen.blit(font.render(f"Score: {score}", True, BLACK), (WIDTH - 160, 20))
    
    # Live HUD Prediction Display
    status_color = GREEN if ("open" in current_pred or "1" in current_pred) else RED
    label_txt = font.render(f"ViT Prediction: {current_pred}", True, status_color)
    screen.blit(label_txt, (20, 20))

    if game_over:
        screen.blit(font.render("GAME OVER! Press 'R' to Restart", True, RED), (WIDTH // 2 - 180, HEIGHT // 2 - 20))

    pygame.display.flip()

pygame.quit()