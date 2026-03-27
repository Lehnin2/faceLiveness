# app/liveness/detector.py - Liveness Detection Logic
import cv2
import mediapipe as mp
import random
import time
import numpy as np
from .ai_model import AntiSpoofingModel

class LivenessDetector:
    def __init__(self):
        # Setup MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            refine_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmark indices
        self.LEFT_EYE_CORNERS = [33, 133]
        self.RIGHT_EYE_CORNERS = [362, 263]
        self.LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Initialize AI model
        self.ai_model = AntiSpoofingModel()
        
        # Challenge configuration
        self.challenges = ["blink", "look_left", "look_right"]
        
        # ADD THESE NEW ATTRIBUTES FOR TIMESTAMP MANAGEMENT
        self.last_process_time = 0
        self.min_frame_interval = 0.033  # ~30 FPS limit (33ms between frames)
        
        self.reset_session()
        
    def reset_session(self):
        """Reset session for new verification"""
        self.challenge = random.choice(self.challenges)
        self.start_time = time.time()
        self.stable_start = None
        self.challenge_met = False
        self.post_success_start = None
        
        # Calibration variables
        self.calibration_frames = 0
        self.center_ratio_sum = 0
        self.center_ratio_avg = 0.5
        self.CALIBRATION_FRAMES = 30
        
        # AI checking variables
        self.ai_check_interval = 0.5
        self.last_ai_check = 0
        self.ai_results = []
        self.AI_BUFFER_SIZE = 5
        
        # Timeouts
        self.POST_SUCCESS_DISPLAY = 2
        self.CHALLENGE_TIMEOUT = 15
        
        # RESET TIMESTAMP TRACKING
        self.last_process_time = 0
        
    def detect_blink_stable(self, landmarks, threshold=0.02):
        """Detect stable blink using eyelid landmarks"""
        left_ratio = abs(landmarks[145].y - landmarks[159].y)
        right_ratio = abs(landmarks[386].y - landmarks[374].y)
        avg_ratio = (left_ratio + right_ratio) / 2
        return avg_ratio < threshold
        
    def get_eye_center(self, landmarks, eye_points):
        """Calculate the center of the eye using multiple eye landmarks"""
        center_x = sum([landmarks[i].x for i in eye_points]) / len(eye_points)
        center_y = sum([landmarks[i].y for i in eye_points]) / len(eye_points)
        return center_x, center_y
        
    def get_gaze_ratio(self, landmarks, eye_points, eye_corners):
        """Calculate gaze direction ratio using eye center relative to eye corners"""
        center_x, center_y = self.get_eye_center(landmarks, eye_points)
        left_corner_x = landmarks[eye_corners[0]].x
        right_corner_x = landmarks[eye_corners[1]].x
        
        eye_width = right_corner_x - left_corner_x
        if eye_width == 0:
            return 0.5
            
        ratio = (center_x - left_corner_x) / eye_width
        return max(0.0, min(1.0, ratio))
        
    def extract_face_region(self, frame, landmarks):
        """Extract face region from frame using landmarks"""
        h, w = frame.shape[:2]
        
        face_points = []
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            face_points.append((x, y))
        
        xs = [p[0] for p in face_points]
        ys = [p[1] for p in face_points]
        
        x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
        y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 20)
        
        face_roi = frame[y_min:y_max, x_min:x_max]
        return face_roi
        
    def process_frame(self, frame):
        """
        Process a single frame for liveness detection
        Returns: (status, message, is_complete, processed_frame)
        """
        current_time = time.time()
        
        # ADD FRAME RATE THROTTLING TO PREVENT TIMESTAMP ISSUES
        if current_time - self.last_process_time < self.min_frame_interval:
            # Return last known state if processing too fast
            return "processing", "Processing...", False, frame
        
        self.last_process_time = current_time
        
        # WRAP MEDIAPIPE PROCESSING IN TRY-CATCH FOR TIMESTAMP ERRORS
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(frame_rgb)
        except Exception as e:
            print(f"MediaPipe processing error: {str(e)}")
            return "error", "Processing error", False, frame
        
        status = "processing"
        message = ""
        is_complete = False
        
        if not result.multi_face_landmarks:
            message = "No face detected"
            return status, message, is_complete, frame
            
        landmarks = result.multi_face_landmarks[0].landmark
        
        # Periodic AI anti-spoofing check
        if current_time - self.last_ai_check > self.ai_check_interval:
            try:
                face_roi = self.extract_face_region(frame, landmarks)
                if face_roi.size > 0:
                    is_real, confidence = self.ai_model.predict(face_roi)
                    self.ai_results.append((is_real, confidence))
                    
                    if len(self.ai_results) > self.AI_BUFFER_SIZE:
                        self.ai_results.pop(0)
                        
                self.last_ai_check = current_time
            except Exception as e:
                print(f"AI model error: {str(e)}")
                # Continue processing even if AI model fails
            
        # Check if AI consistently says it's real
        ai_says_real = True
        if self.ai_results:
            real_count = sum(1 for is_real, _ in self.ai_results if is_real)
            ai_says_real = real_count >= len(self.ai_results) * 0.6
            
        # Calculate gaze ratios
        left_ratio = self.get_gaze_ratio(landmarks, self.LEFT_EYE_POINTS, self.LEFT_EYE_CORNERS)
        right_ratio = self.get_gaze_ratio(landmarks, self.RIGHT_EYE_POINTS, self.RIGHT_EYE_CORNERS)
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Calibration phase
        if self.calibration_frames < self.CALIBRATION_FRAMES:
            self.center_ratio_sum += avg_ratio
            self.calibration_frames += 1
            if self.calibration_frames == self.CALIBRATION_FRAMES:
                self.center_ratio_avg = self.center_ratio_sum / self.CALIBRATION_FRAMES
            message = f"Calibrating... Look straight ahead ({self.calibration_frames}/{self.CALIBRATION_FRAMES})"
        else:
            # Challenge detection
            if not self.challenge_met and ai_says_real:
                deviation = avg_ratio - self.center_ratio_avg
                
                if self.challenge == "blink":
                    if self.detect_blink_stable(landmarks):
                        if self.stable_start is None:
                            self.stable_start = current_time
                        elif current_time - self.stable_start >= 0.5:
                            self.challenge_met = True
                            self.post_success_start = current_time
                            status = "success"
                            message = "✅ Blink detected successfully!"
                    else:
                        self.stable_start = None
                        
                elif self.challenge == "look_right":
                    if deviation > 0.08:
                        if self.stable_start is None:
                            self.stable_start = current_time
                        elif current_time - self.stable_start >= 0.8:
                            self.challenge_met = True
                            self.post_success_start = current_time
                            status = "success"
                            message = "✅ Look right detected successfully!"
                    else:
                        self.stable_start = None
                        
                elif self.challenge == "look_left":
                    if deviation < -0.08:
                        if self.stable_start is None:
                            self.stable_start = current_time
                        elif current_time - self.stable_start >= 0.8:
                            self.challenge_met = True
                            self.post_success_start = current_time
                            status = "success"
                            message = "✅ Look left detected successfully!"
                    else:
                        self.stable_start = None
                        
            elif not ai_says_real:
                self.stable_start = None
                status = "spoofing"
                message = "❌ Spoofing detected!"
                
            if not self.challenge_met and not message:
                message = f"Do this: {self.challenge.replace('_', ' ')}"
                
        # Check for completion or timeout
        if self.challenge_met:
            if current_time - self.post_success_start > self.POST_SUCCESS_DISPLAY:
                is_complete = True
                status = "verified"
        elif (self.calibration_frames >= self.CALIBRATION_FRAMES and 
              current_time - self.start_time > self.CHALLENGE_TIMEOUT):
            is_complete = True
            status = "failed"
            message = "❌ Challenge failed - timeout!"
            
        # WRAP DRAWING IN TRY-CATCH TO PREVENT ADDITIONAL ERRORS
        try:
            # Draw face mesh on frame
            self.mp_drawing.draw_landmarks(
                frame, 
                result.multi_face_landmarks[0], 
                self.mp_face_mesh.FACEMESH_CONTOURS
            )
        except Exception as e:
            print(f"Drawing error: {str(e)}")
            # Continue without drawing if there's an error
        
        return status, message, is_complete, frame
        
    def get_current_challenge(self):
        """Get the current challenge"""
        return self.challenge
        
    def get_ai_status(self):
        """Get current AI spoofing detection status"""
        if not self.ai_results:
            return "Checking...", (255, 255, 255)
            
        real_count = sum(1 for is_real, _ in self.ai_results if is_real)
        if real_count >= len(self.ai_results) * 0.6:
            return "✅ AI: REAL", (0, 255, 0)
        else:
            return "❌ AI: SPOOF", (0, 0, 255)