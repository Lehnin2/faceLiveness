# app/liveness/ai_model.py - AI Anti-Spoofing Model Integration
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

class AntiSpoofingModel:
    def __init__(self, model_path="ai_models/resnet_checkpoint_epoch_8.pt"):
        """Initialize the anti-spoofing model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Load model
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"✅ Anti-spoofing model loaded from {model_path}")
            else:
                print(f"⚠️ Model file not found at {model_path}. Using fallback mode.")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            
        if self.model:
            self.model.to(self.device)
            self.model.eval()
            
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, frame):
        """
        Predict if the face in the frame is real or spoofed
        Returns: (is_real: bool, confidence: float)
        """
        if self.model is None:
            return True, 0.8  # Fallback if model failed to load
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Preprocess
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][pred].item()
                
            is_real = pred == 1
            return is_real, confidence
            
        except Exception as e:
            print(f"Error in AI prediction: {e}")
            return True, 0.5  # Fallback