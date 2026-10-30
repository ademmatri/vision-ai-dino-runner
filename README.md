# ViT Gesture-Controlled Dino Runner & Computer Vision Suite

A real-time Vision Transformer (ViT) application built with PyTorch, Transformers, OpenCV, and Pygame. Control a Dino runner game with hand gestures captured by your webcam!

## Repository Structure
- \dinoGame.py\ - Real-time Dino runner game controlled via fine-tuned ViT gesture predictions.
- \	rain.py\ - Script to fine-tune the Vision Transformer model on custom dataset.
- \webcam_predict.py\ - Standalone real-time webcam inference pipeline.
- \predict.py\ - Single image evaluation script.
- \pp.py\ - Gradio web interface for model interactive demo.

## Quick Start

1. Clone the repository:
   ash
   git clone https://github.com/ademmatri/vision-ai-dino-runner.git
   cd vision-ai-dino-runner
   

3. **Install dependencies:**
   ash
   pip install -r requirements.txt
   

4. **Run the Game:**
   ash
   python dinoGame.py
   

## Controls
- **Open Hand:** Jump!
- **Closed Fist:** Run / Reset jump trigger.
- **Press 'R':** Restart after Game Over.
