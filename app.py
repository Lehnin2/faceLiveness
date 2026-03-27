#correct parameters
import cv2
import mediapipe as mp
import random
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices
LEFT_EYE_CORNERS = [33, 133]   # left corner, right corner of left eye
RIGHT_EYE_CORNERS = [362, 263] # left corner, right corner of right eye

# Eye landmarks for calculating center (pupil approximation)
LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Challenge list
challenges = ["blink", "look_left", "look_right"]

class AntiSpoofingModel:
    def __init__(self, model_path="resnet_checkpoint_epoch_8.pt"):
        """Initialize the anti-spoofing model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Anti-spoofing model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            return
            
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
            return True, 0.0  # Fallback if model failed to load
            
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
            return True, 0.0  # Fallback

def detect_blink_stable(landmarks, threshold=0.02):
    # Use proper eyelid landmarks for blink detection
    left_ratio = abs(landmarks[145].y - landmarks[159].y)
    right_ratio = abs(landmarks[386].y - landmarks[374].y)
    avg_ratio = (left_ratio + right_ratio) / 2
    return avg_ratio < threshold

def get_eye_center(landmarks, eye_points):
    """Calculate the center of the eye using multiple eye landmarks"""
    center_x = sum([landmarks[i].x for i in eye_points]) / len(eye_points)
    center_y = sum([landmarks[i].y for i in eye_points]) / len(eye_points)
    return center_x, center_y

def get_gaze_ratio(landmarks, eye_points, eye_corners):
    """Calculate gaze direction ratio using eye center relative to eye corners"""
    # Get eye center (approximates pupil position)
    center_x, center_y = get_eye_center(landmarks, eye_points)
    
    # Get eye corner positions
    left_corner_x = landmarks[eye_corners[0]].x
    right_corner_x = landmarks[eye_corners[1]].x
    
    # Calculate eye width
    eye_width = right_corner_x - left_corner_x
    if eye_width == 0:
        return 0.5
    
    # Calculate ratio (0 = looking left, 1 = looking right)
    ratio = (center_x - left_corner_x) / eye_width
    return max(0.0, min(1.0, ratio))

def extract_face_region(frame, landmarks):
    """Extract face region from frame using landmarks"""
    h, w = frame.shape[:2]
    
    # Get face boundary points
    face_points = []
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        face_points.append((x, y))
    
    # Get bounding box
    xs = [p[0] for p in face_points]
    ys = [p[1] for p in face_points]
    
    x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
    y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 20)
    
    # Extract face region
    face_roi = frame[y_min:y_max, x_min:x_max]
    return face_roi

# Initialize AI model
print("🤖 Loading Anti-Spoofing AI Model...")
ai_model = AntiSpoofingModel()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

challenge = random.choice(challenges)
start_time = time.time()

# For stable detection
stable_start = None
challenge_met = False
post_success_start = None
POST_SUCCESS_DISPLAY = 2  # seconds to display success message before closing
CHALLENGE_TIMEOUT = 10    # seconds before challenge fails

print(f"Challenge: {challenge}")

# Calibration variables
calibration_frames = 0
center_ratio_sum = 0
center_ratio_avg = 0.5  # Default center
CALIBRATION_FRAMES = 30  # Number of frames to calibrate

# AI checking variables
ai_check_interval = 0.5  # Check AI every 0.5 seconds
last_ai_check = 0
ai_results = []  # Store recent AI results
AI_BUFFER_SIZE = 5  # Number of AI results to keep for averaging

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    current_time = time.time()
    
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Periodic AI anti-spoofing check
        if current_time - last_ai_check > ai_check_interval:
            face_roi = extract_face_region(frame, landmarks)
            if face_roi.size > 0:
                is_real, confidence = ai_model.predict(face_roi)
                ai_results.append((is_real, confidence))
                
                # Keep only recent results
                if len(ai_results) > AI_BUFFER_SIZE:
                    ai_results.pop(0)
                
                print(f"🤖 AI Check: {'REAL' if is_real else 'SPOOF'} (confidence: {confidence:.3f})")
                
            last_ai_check = current_time

        # Check if AI consistently says it's real
        ai_says_real = True
        if ai_results:
            real_count = sum(1 for is_real, _ in ai_results if is_real)
            ai_says_real = real_count >= len(ai_results) * 0.6  # 60% threshold

        # Calculate current gaze ratios
        left_ratio = get_gaze_ratio(landmarks, LEFT_EYE_POINTS, LEFT_EYE_CORNERS)
        right_ratio = get_gaze_ratio(landmarks, RIGHT_EYE_POINTS, RIGHT_EYE_CORNERS)
        avg_ratio = (left_ratio + right_ratio) / 2

        # Calibration phase - collect center position data
        if calibration_frames < CALIBRATION_FRAMES:
            center_ratio_sum += avg_ratio
            calibration_frames += 1
            if calibration_frames == CALIBRATION_FRAMES:
                center_ratio_avg = center_ratio_sum / CALIBRATION_FRAMES
                print(f"Calibration complete! Center ratio: {center_ratio_avg:.3f}")
        else:
            # Only check challenge if not already met, calibration is done, AND AI says it's real
            if not challenge_met and ai_says_real:
                # Calculate deviation from center
                deviation = avg_ratio - center_ratio_avg
                
                # Debug output
                print(f"Gaze ratios - Left eye: {left_ratio:.3f}, Right eye: {right_ratio:.3f}, Average: {avg_ratio:.3f}, Deviation: {deviation:.3f}")
                
                if challenge == "blink":
                    if detect_blink_stable(landmarks):
                        if stable_start is None:
                            stable_start = current_time
                        elif current_time - stable_start >= 0.5:  # held closed for 0.5s
                            challenge_met = True
                            post_success_start = current_time
                            print("✅ Blink detected successfully!")
                    else:
                        stable_start = None

                elif challenge == "look_right":
                    # Looking right means deviation should be positive and significant
                    if deviation > 0.08:  # Adjust threshold based on your testing
                        if stable_start is None:
                            stable_start = current_time
                        elif current_time - stable_start >= 0.8:  # Hold for 0.8 seconds
                            challenge_met = True
                            post_success_start = current_time
                            print("✅ Look right detected successfully!")
                    else:
                        stable_start = None

                elif challenge == "look_left":
                    # Looking left means deviation should be negative and significant
                    if deviation < -0.08:  # Adjust threshold based on your testing
                        if stable_start is None:
                            stable_start = current_time
                        elif current_time - stable_start >= 0.8:  # Hold for 0.8 seconds
                            challenge_met = True
                            post_success_start = current_time
                            print("✅ Look left detected successfully!")
                    else:
                        stable_start = None
            elif not ai_says_real:
                # Reset challenge progress if AI detects spoofing
                stable_start = None

        # Draw face mesh
        mp_drawing.draw_landmarks(frame, result.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)

    # Display AI status
    if ai_results:
        real_count = sum(1 for is_real, _ in ai_results if is_real)
        ai_status = "✅ AI: REAL" if real_count >= len(ai_results) * 0.6 else "❌ AI: SPOOF"
        color = (0, 255, 0) if real_count >= len(ai_results) * 0.6 else (0, 0, 255)
        cv2.putText(frame, ai_status, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Instructions
    if calibration_frames < CALIBRATION_FRAMES:
        cv2.putText(frame, f"Calibrating... Look straight ahead ({calibration_frames}/{CALIBRATION_FRAMES})", 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"Do this: {challenge.replace('_', ' ')}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display result and handle exit conditions
    if challenge_met:
        cv2.putText(frame, "✅ VERIFIED REAL USER", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        # Exit after displaying success message
        if current_time - post_success_start > POST_SUCCESS_DISPLAY:
            print("🎉 Challenge completed successfully! User verified as real.")
            break
    elif calibration_frames >= CALIBRATION_FRAMES and current_time - start_time > CHALLENGE_TIMEOUT:
        if not ai_says_real:
            cv2.putText(frame, "❌ SPOOFING DETECTED", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "❌ CHALLENGE FAILED", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # Exit 2 seconds after showing failure message
        if current_time - start_time > CHALLENGE_TIMEOUT + 2:
            print("❌ Challenge failed - timeout or spoofing detected!")
            break

    cv2.imshow("Advanced Liveness Detection", frame)

    # ESC key to quit manually
    if cv2.waitKey(5) & 0xFF == 27:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()