from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import mediapipe as mp
import random
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance
import numpy as np
import base64
import io
import sqlite3
import hashlib
import jwt
import datetime
from functools import wraps
import os
import face_recognition
import json
import pytesseract
import re
import time
import uuid
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from io import BytesIO


# Configure Tesseract path if needed (uncomment and adjust path for your system)
# Windows example:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            face_encoding TEXT,
            registration_date TEXT,
            last_login TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()


# Create directory for storing face images
FACE_IMAGES_DIR = "static/faces"
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)

def save_face_image_and_encoding(username, base64_image_data):
    """
    Save both the face image as a file AND the face encoding in the database
    This dual approach provides redundancy and better matching
    """
    try:
        # Convert base64 to image
        if ',' in base64_image_data:
            base64_image_data = base64_image_data.split(',')[1]
        
        image_data = base64.b64decode(base64_image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, "Failed to decode image"
        
        # Save image file
        image_filename = f"{username}_face.jpg"
        image_path = os.path.join(FACE_IMAGES_DIR, image_filename)
        cv2.imwrite(image_path, image)
        print(f"✅ Face image saved: {image_path}")
        
        # Extract face encoding
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)
        
        if len(face_encodings) == 0:
            return None, "No face detected in image"
        
        face_encoding = face_encodings[0]
        
        # Store both in database
        face_data = {
            'encoding': face_encoding.tolist(),
            'image_path': image_path,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return json.dumps(face_data), None
        
    except Exception as e:
        print(f"Error in save_face_image_and_encoding: {e}")
        return None, str(e)

def load_face_encoding(face_data_json):
    """
    Load face encoding from stored JSON data with fallback to image file
    """
    try:
        face_data = json.loads(face_data_json)
        
        # Try to load from encoding first (faster)
        if 'encoding' in face_data:
            encoding = np.array(face_data['encoding'])
            print(f"✅ Loaded face encoding from JSON (shape: {encoding.shape})")
            return encoding, "encoding"
        
        # Fallback to loading from image file
        if 'image_path' in face_data and os.path.exists(face_data['image_path']):
            image = cv2.imread(face_data['image_path'])
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)
            
            if len(face_encodings) > 0:
                print(f"✅ Loaded face encoding from image file")
                return face_encodings[0], "image_file"
        
        return None, "no_valid_data"
        
    except Exception as e:
        print(f"Error loading face encoding: {e}")
        return None, "error"

def enhanced_face_compare(live_frame, stored_face_data, tolerance=0.6):
    """
    Enhanced face comparison with multiple strategies and detailed logging
    """
    try:
        print(f"🔍 Starting face comparison with tolerance: {tolerance}")
        
        # Load stored face encoding
        stored_encoding, load_method = load_face_encoding(stored_face_data)
        if stored_encoding is None:
            print(f"❌ Failed to load stored face encoding")
            return False, 0.0
        
        print(f"📊 Stored encoding loaded via: {load_method}")
        print(f"📏 Stored encoding shape: {stored_encoding.shape}")
        
        # Extract face from live frame
        rgb_frame = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
        
        # Try both HOG and CNN models for better detection
        live_encodings_hog = face_recognition.face_encodings(rgb_frame, model="hog")
        live_encodings_cnn = face_recognition.face_encodings(rgb_frame, model="cnn")
        
        print(f"👥 Faces detected - HOG: {len(live_encodings_hog)}, CNN: {len(live_encodings_cnn)}")
        
        if len(live_encodings_hog) == 0 and len(live_encodings_cnn) == 0:
            print("❌ No face detected in live frame")
            return False, 0.0
        
        # Use the encoding from whichever method detected a face
        live_encoding = live_encodings_cnn[0] if len(live_encodings_cnn) > 0 else live_encodings_hog[0]
        detection_method = "CNN" if len(live_encodings_cnn) > 0 else "HOG"
        
        print(f"🎯 Using {detection_method} detection")
        print(f"📏 Live encoding shape: {live_encoding.shape}")
        
        # Calculate face distance
        face_distance = face_recognition.face_distance([stored_encoding], live_encoding)[0]
        confidence = 1 - face_distance
        is_match = face_distance <= tolerance
        
        print(f"📊 Face distance: {face_distance:.4f}")
        print(f"📈 Confidence: {confidence:.4f}")
        print(f"✅ Match result: {is_match} (threshold: {tolerance})")
        
        # Additional quality checks
        if is_match:
            print("🎉 Face match successful!")
        else:
            print(f"❌ Face match failed (distance {face_distance:.4f} > threshold {tolerance})")
            
            # Suggest different thresholds for debugging
            for test_tolerance in [0.4, 0.5, 0.7, 0.8]:
                would_match = face_distance <= test_tolerance
                print(f"   With tolerance {test_tolerance}: {'✅ MATCH' if would_match else '❌ NO MATCH'}")
        
        return is_match, confidence
        
    except Exception as e:
        print(f"❌ Error in face comparison: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0

def migrate_existing_face_data():
    """
    Migrate existing base64 image data to the new format
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT username, face_encoding FROM users WHERE face_encoding IS NOT NULL')
        users = cursor.fetchall()
        
        for username, face_data in users:
            print(f"🔄 Migrating user: {username}")
            
            # Check if it's already in new format
            if face_data.startswith('{'):
                print(f"   Already in new format")
                continue
            
            # Convert base64 image to new format
            new_face_data, error = save_face_image_and_encoding(username, face_data)
            
            if new_face_data:
                cursor.execute('UPDATE users SET face_encoding = ? WHERE username = ?', 
                             (new_face_data, username))
                print(f"   ✅ Migrated successfully")
            else:
                print(f"   ❌ Migration failed: {error}")
        
        conn.commit()
        print("🎉 Migration complete!")
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
    finally:
        conn.close()

def test_face_matching_system():
    """
    Test the face matching system with sample data
    """
    print("🧪 TESTING ENHANCED FACE MATCHING SYSTEM")
    print("=" * 50)
    
    # Test with a sample user (you can modify this to test with your actual users)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username, face_encoding FROM users WHERE face_encoding IS NOT NULL LIMIT 1')
    user_data = cursor.fetchone()
    conn.close()
    
    if not user_data:
        print("❌ No users with face data found for testing")
        return
    
    username, face_data = user_data
    print(f"🧪 Testing with user: {username}")
    
    # Load the stored face encoding
    stored_encoding, load_method = load_face_encoding(face_data)
    if stored_encoding is not None:
        print(f"✅ Successfully loaded face encoding via: {load_method}")
        print(f"📊 Encoding stats - Min: {stored_encoding.min():.3f}, Max: {stored_encoding.max():.3f}")
        
        # Test self-comparison (should be perfect match)
        if load_method == "image_file":
            face_data_dict = json.loads(face_data)
            if 'image_path' in face_data_dict and os.path.exists(face_data_dict['image_path']):
                test_image = cv2.imread(face_data_dict['image_path'])
                match_result, confidence = enhanced_face_compare(test_image, face_data, tolerance=0.6)
                print(f"🎯 Self-comparison result: Match={match_result}, Confidence={confidence:.4f}")
    else:
        print("❌ Failed to load face encoding for testing")

# Update your Flask route to use the new system
def register_with_enhanced_face_storage(username, email, password, face_image_data):
    """
    Updated registration function using the enhanced face storage system
    """
    try:
        # Check if user already exists
        if get_user_by_username(username):
            return False, 'Username already exists'
        
        if get_user_by_email(email):
            return False, 'Email already registered'
        
        # Save face data using enhanced system
        face_data, error = save_face_image_and_encoding(username, face_image_data)
        
        if not face_data:
            return False, f'Face processing failed: {error}'
        
        # Create user with enhanced face data
        password_hash = hash_password(password)
        success = create_user(username, email, password_hash, face_data)
        
        return success, 'Registration successful' if success else 'Database error'
        
    except Exception as e:
        return False, str(e)
    

def check_face_duplicate(new_face_data, exclude_username=None):
    """
    Check if the provided face encoding matches any existing user's face
    Returns (is_duplicate, matching_username, confidence) if duplicate found
    """
    try:
        print("🔍 Checking for face duplicates...")
        
        # Get all users with face data
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        if exclude_username:
            cursor.execute('SELECT username, face_encoding FROM users WHERE face_encoding IS NOT NULL AND username != ?', (exclude_username,))
        else:
            cursor.execute('SELECT username, face_encoding FROM users WHERE face_encoding IS NOT NULL')
        
        existing_users = cursor.fetchall()
        conn.close()
        
        print(f"📊 Found {len(existing_users)} users with face data to check against")
        
        if len(existing_users) == 0:
            print("✅ No existing face data to compare against")
            return False, None, 0.0
        
        # Create a temporary frame for the new face to compare
        # Convert base64 to image
        if ',' in new_face_data:
            new_face_data = new_face_data.split(',')[1]
        
        image_data = base64.b64decode(new_face_data)
        nparr = np.frombuffer(image_data, np.uint8)
        new_face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if new_face_image is None:
            print("❌ Failed to decode new face image")
            return False, None, 0.0
        
        # Check against each existing user
        for username, stored_face_data in existing_users:
            print(f"🔍 Comparing against user: {username}")
            
            try:
                # Use enhanced face comparison
                is_match, confidence = enhanced_face_compare(new_face_image, stored_face_data, tolerance=0.6)
                
                print(f"   Match result: {is_match}, Confidence: {confidence:.3f}")
                
                # Use a stricter threshold for duplicate detection (higher confidence required)
                if is_match and confidence > 0.65:  # Stricter than login threshold
                    print(f"🚨 DUPLICATE DETECTED! User {username} has matching face (confidence: {confidence:.3f})")
                    return True, username, confidence
                
                # Also check with slightly relaxed threshold for edge cases
                if confidence > 0.55:  # Still high confidence but not exact match
                    print(f"⚠️  Potential duplicate with {username} (confidence: {confidence:.3f})")
                    # You might want to flag this for manual review
                    
            except Exception as e:
                print(f"❌ Error comparing with user {username}: {e}")
                continue
        
        print("✅ No face duplicates found")
        return False, None, 0.0
        
    except Exception as e:
        print(f"❌ Error in face duplicate check: {e}")
        import traceback
        traceback.print_exc()
        return False, None, 0.0
# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Challenge list
challenges = ["blink", "look_left", "look_right"]

def compare_faces(live_frame, stored_face_data):
    """
    Main face comparison function for your Flask app
    """
    return enhanced_face_compare(live_frame, stored_face_data, tolerance=0.6)

if __name__ == "__main__":
    # Run migration for existing users
    print("🔄 Running face data migration...")
    migrate_existing_face_data()
    
    # Test the system
    test_face_matching_system()

class AntiSpoofingModel:
    def __init__(self, model_path="models/resnet_checkpoint_epoch_8.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
        try:
            if os.path.exists(model_path):
                print(f"📁 Model file found: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
                
                # Check if the checkpoint has the expected structure
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"✅ Model loaded from checkpoint")
                else:
                    print("⚠️ No 'model_state_dict' key found, trying direct load")
                    self.model.load_state_dict(checkpoint)
                
                print(f"✅ Anti-spoofing model loaded successfully")
            else:
                print(f"❌ Model file not found: {model_path}")
                print(f"📂 Current working directory: {os.getcwd()}")
                print(f"📂 Files in directory: {os.listdir('.')}")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
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
        if self.model is None:
            print("⚠️ No model loaded - returning dummy values")
            return random.choice([True, False]), random.uniform(0.7, 0.95)
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Debug: Check image size
            print(f"🖼️ Input image size: {pil_image.size}")
            
            # Preprocess
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            print(f"📊 Tensor shape: {img_tensor.shape}")
            
            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][pred].item()
                
                print(f"🎯 Raw output: {output}")
                print(f"📈 Probabilities: {probabilities}")
                print(f"🎲 Prediction: {pred} (0=spoof, 1=real), Confidence: {confidence:.3f}")
                
            is_real = pred == 1
            return is_real, confidence
            
        except Exception as e:
            print(f"❌ Error in AI prediction: {e}")
            import traceback
            traceback.print_exc()
            return True, 0.5  # Default to real with low confidence


# Initialize AI model
ai_model = AntiSpoofingModel()

# Liveness detection functions
def detect_blink_stable(landmarks, threshold=0.02):
    left_ratio = abs(landmarks[145].y - landmarks[159].y)
    right_ratio = abs(landmarks[386].y - landmarks[374].y)
    avg_ratio = (left_ratio + right_ratio) / 2
    print(f"Blink ratios - Left: {left_ratio:.4f}, Right: {right_ratio:.4f}, Avg: {avg_ratio:.4f}, Threshold: {threshold}")
    return avg_ratio < threshold

def get_eye_center(landmarks, eye_points):
    """Calculate the center of the eye using multiple eye landmarks"""
    center_x = sum([landmarks[i].x for i in eye_points]) / len(eye_points)
    center_y = sum([landmarks[i].y for i in eye_points]) / len(eye_points)
    return center_x, center_y

def get_gaze_ratio(landmarks, eye_points, eye_corners):
    """Calculate gaze direction ratio using eye center relative to eye corners"""
    center_x, center_y = get_eye_center(landmarks, eye_points)
    
    left_corner_x = landmarks[eye_corners[0]].x
    right_corner_x = landmarks[eye_corners[1]].x
    
    eye_width = right_corner_x - left_corner_x
    if eye_width == 0:
        return 0.5
    
    ratio = (center_x - left_corner_x) / eye_width
    return max(0.0, min(1.0, ratio))

def extract_face_region(frame, landmarks):
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

# Utility functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token(username):
    payload = {
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['username']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            # Handle both "Bearer token" and just "token" formats
            if token.startswith('Bearer '):
                token = token.split(' ')[1]
            username = verify_token(token)
            if not username:
                return jsonify({'message': 'Token is invalid'}), 401
        except Exception as e:
            print(f"Token verification error: {e}")
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(username, *args, **kwargs)
    return decorated

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

# Database functions
def get_user_by_username(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_by_email(email):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(username, email, password_hash, face_encoding=None):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, face_encoding, registration_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, face_encoding, datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def update_user_face(username, face_encoding):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET face_encoding = ? WHERE username = ?', 
                   (face_encoding, username))
    conn.commit()
    conn.close()

def update_last_login(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET last_login = ? WHERE username = ?', 
                   (datetime.datetime.now().isoformat(), username))
    conn.commit()
    conn.close()

# Store active liveness sessions
liveness_sessions = {}

# In your LivenessSession class, modify the __init__ method:
class LivenessSession:
    def __init__(self, username):
        self.username = username
        self.ai_check_interval = 0.5
        self.AI_BUFFER_SIZE = 5
        self.challenge = random.choice(challenges)
        self.start_time = time.time()
        self.calibration_frames = 0
        self.center_ratio_sum = 0
        self.center_ratio_avg = 0.5
        self.stable_start = None
        self.challenge_met = False
        self.ai_results = []
        self.last_ai_check = 0
        self.status = "calibrating"
        self.message = "Look straight ahead for calibration"
        self.post_success_start = None
        self.CALIBRATION_FRAMES = 30
        self.CHALLENGE_TIMEOUT = 15  # Increased timeout to 15 seconds
        self.POST_SUCCESS_DISPLAY = 2
        self.challenge_start_time = None  # NEW: Track when challenge phase starts
        
    def to_dict(self):
        # Calculate time remaining for challenge phase
        if self.challenge_start_time and self.status == "challenging":
            time_remaining = max(0, self.CHALLENGE_TIMEOUT - (time.time() - self.challenge_start_time))
        else:
            time_remaining = self.CHALLENGE_TIMEOUT
            
        return {
            'challenge': self.challenge,
            'status': self.status,
            'message': self.message,
            'calibration_progress': min(self.calibration_frames, self.CALIBRATION_FRAMES) / self.CALIBRATION_FRAMES,
            'time_remaining': time_remaining
        }

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        face_image_data = data.get('face_encoding')  # This is actually base64 image data
        
        if not username or not email or not password or not face_image_data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user already exists
        if get_user_by_username(username):
            return jsonify({'error': 'Username already exists'}), 409
        
        if get_user_by_email(email):
            return jsonify({'error': 'Email already registered'}), 409
        
        # NEW: Check for face duplicates BEFORE processing
        print("🔍 Checking for face duplicates before registration...")
        is_duplicate, existing_username, confidence = check_face_duplicate(face_image_data)
        
        if is_duplicate:
            print(f"🚨 Registration blocked - face already registered to user: {existing_username}")
            return jsonify({
                'error': f'This face is already registered to another account. Each person can only have one account.',
                'error_type': 'duplicate_face',
                'existing_user': existing_username,  # Don't include this in production for privacy
                'confidence': confidence
            }), 409  # Conflict status code
        
        # Use enhanced face storage system
        face_data, error = save_face_image_and_encoding(username, face_image_data)
        
        if not face_data:
            return jsonify({'error': f'Face processing failed: {error}'}), 400
        
        # Create user with enhanced face data
        password_hash = hash_password(password)
        if create_user(username, email, password_hash, face_data):
            print(f"✅ User {username} registered with enhanced face data")
            return jsonify({'message': 'User registered successfully with enhanced face data'}), 201
        else:
            return jsonify({'error': 'Registration failed'}), 500
            
    except Exception as e:
        print(f"Registration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Missing username or password'}), 400
        
        user = get_user_by_username(username)
        if not user or user[3] != hash_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate token
        token = generate_token(username)
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'has_face': user[4] is not None
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-face', methods=['POST'])
@token_required
def save_face(username):
    try:
        data = request.get_json()
        face_data = data.get('face_data')
        
        if not face_data:
            return jsonify({'error': 'No face data provided'}), 400
        
        # Check for duplicates (excluding current user)
        is_duplicate, existing_username, confidence = check_face_duplicate(face_data, exclude_username=username)
        
        if is_duplicate:
            return jsonify({
                'error': 'This face is already registered to another account',
                'error_type': 'duplicate_face'
            }), 409
        
        # Save face encoding
        update_user_face(username, face_data)
        
        return jsonify({'message': 'Face saved successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-liveness', methods=['POST'])
@token_required
def start_liveness(username):
    try:
        # Create new liveness session
        session = LivenessSession(username)
        liveness_sessions[username] = session
        
        return jsonify({
            'message': 'Liveness detection started',
            'session_data': session.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
def update_last_login(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE users 
            SET last_login = ? 
            WHERE username = ?
        ''', (datetime.datetime.now().isoformat(), username))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating last login: {e}")
        return False
    finally:
        conn.close()

@app.route('/api/check-liveness', methods=['POST'])
@token_required
def check_liveness(username):
    try:
        print(f"Liveness check for user: {username}")
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        session = liveness_sessions.get(username)
        if not session:
            print(f"No session found for {username}")
            return jsonify({'error': 'No active liveness session'}), 400

        frame = base64_to_image(frame_data)
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        
        current_time = time.time()
        ai_says_real = True
        
        # Add new session variables if they don't exist
        if not hasattr(session, 'challenge_start_ai_checked'):
            session.challenge_start_ai_checked = False
        if not hasattr(session, 'challenge_complete_ai_checked'):
            session.challenge_complete_ai_checked = False
        
        def run_multiple_ai_predictions(frame, landmarks, num_predictions=3):
            """Run multiple AI predictions and return average result"""
            predictions = []
            confidences = []
            
            face_roi = extract_face_region(frame, landmarks)
            if face_roi.size > 0:
                for i in range(num_predictions):
                    is_real, confidence = ai_model.predict(face_roi)
                    predictions.append(is_real)
                    confidences.append(confidence)
                    print(f"🤖 AI Prediction {i+1}/{num_predictions}: Real={is_real}, Confidence={confidence:.3f}")
                
                # Calculate weighted average
                real_count = sum(predictions)
                avg_confidence = sum(confidences) / len(confidences)
                final_is_real = real_count >= (num_predictions / 2)  # Majority vote
                
                print(f"🎯 AI Final Result: {real_count}/{num_predictions} real, Avg Confidence: {avg_confidence:.3f}, Final: {final_is_real}")
                return final_is_real, avg_confidence
            
            return True, 0.5  # Default if no face ROI
        
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            
            # Calculate gaze ratios
            left_ratio = get_gaze_ratio(landmarks, LEFT_EYE_POINTS, LEFT_EYE_CORNERS)
            right_ratio = get_gaze_ratio(landmarks, RIGHT_EYE_POINTS, RIGHT_EYE_CORNERS)
            avg_ratio = (left_ratio + right_ratio) / 2
            
            # Calibration phase
            if session.calibration_frames < session.CALIBRATION_FRAMES:
                session.center_ratio_sum += avg_ratio
                session.calibration_frames += 1
                if session.calibration_frames == session.CALIBRATION_FRAMES:
                    session.center_ratio_avg = session.center_ratio_sum / session.CALIBRATION_FRAMES
                    session.status = "challenging"
                    session.message = f"Please {session.challenge.replace('_', ' ')}"
                    session.challenge_start_time = current_time
            else:
                # FIRST AI CHECK - At the start of challenge phase
                if not session.challenge_start_ai_checked:
                    print("🔍 Running FIRST AI check at challenge start...")
                    ai_says_real, ai_confidence = run_multiple_ai_predictions(frame, landmarks)
                    session.challenge_start_ai_checked = True
                    
                    if not ai_says_real and ai_confidence > 0.7:
                        session.status = "failed"
                        session.message = f"Spoofing detected at challenge start (confidence: {ai_confidence:.2f})"
                        return jsonify({
                            'session_data': session.to_dict(),
                            'ai_status': 'spoof_detected',
                            'ai_check_point': 'challenge_start'
                        }), 200
                
                # Challenge phase
                if not session.challenge_met:
                    # Check for timeout
                    if current_time - session.challenge_start_time > session.CHALLENGE_TIMEOUT:
                        session.status = "failed"
                        session.message = f"Challenge timeout - please try again"
                        print(f"❌ Challenge timeout after {session.CHALLENGE_TIMEOUT} seconds")
                        return jsonify({
                            'session_data': session.to_dict()
                        }), 200
                    
                    deviation = avg_ratio - session.center_ratio_avg
                    
                    print(f"Challenge: {session.challenge}")
                    print(f"Center avg: {session.center_ratio_avg:.4f}, Current avg: {avg_ratio:.4f}")
                    print(f"Deviation: {deviation:.4f}")
                    print(f"Time elapsed: {current_time - session.challenge_start_time:.1f}s / {session.CHALLENGE_TIMEOUT}s")
                    
                    challenge_detected = False
                    
                    if session.challenge == "blink":
                        blink_detected = detect_blink_stable(landmarks)
                        print(f"Blink detected: {blink_detected}")
                        if blink_detected:
                            challenge_detected = True
                    elif session.challenge == "look_right":
                        print(f"Look right threshold: 0.08, deviation: {deviation:.4f}")
                        if deviation > 0.08:
                            challenge_detected = True
                            print("✅ Look right detected!")
                    elif session.challenge == "look_left":
                        print(f"Look left threshold: -0.08, deviation: {deviation:.4f}")
                        if deviation < -0.08:
                            challenge_detected = True
                            print("✅ Look left detected!")
                    
                    print(f"Challenge detected: {challenge_detected}")
                    
                    if challenge_detected:
                        if session.stable_start is None:
                            session.stable_start = current_time
                            print("⏱️ Started stability timer")
                        
                        if session.challenge == "blink":
                            hold_time = 0.5
                        else:
                            hold_time = 1.0

                        time_held = current_time - session.stable_start
                        print(f"Holding challenge for {time_held:.2f}s / {hold_time}s required")
                        
                        # Update message to show progress
                        remaining_time = max(0, session.CHALLENGE_TIMEOUT - (current_time - session.challenge_start_time))
                        session.message = f"Hold {session.challenge.replace('_', ' ')} for {hold_time:.1f}s ({time_held:.1f}s) - {remaining_time:.0f}s remaining"
                        
                        if time_held >= hold_time:
                            print("🎉 Challenge completed successfully!")
                            
                            # SECOND AI CHECK - When challenge is completed
                            if not session.challenge_complete_ai_checked:
                                print("🔍 Running SECOND AI check at challenge completion...")
                                ai_says_real, ai_confidence = run_multiple_ai_predictions(frame, landmarks)
                                session.challenge_complete_ai_checked = True
                                
                                if not ai_says_real and ai_confidence > 0.7:
                                    session.status = "failed"
                                    session.message = f"Spoofing detected at challenge completion (confidence: {ai_confidence:.2f})"
                                    return jsonify({
                                        'session_data': session.to_dict(),
                                        'ai_status': 'spoof_detected',
                                        'ai_check_point': 'challenge_complete'
                                    }), 200
                            
                            # Set up return to center phase
                            session.challenge_met = True
                            session.status = "return_center"
                            session.message = "Great! Now look straight ahead for face verification"
                            session.return_center_start = current_time
                    else:
                        # Challenge not detected, reset timer
                        if session.stable_start is not None:
                            print("❌ Challenge lost, resetting stability timer")
                            session.stable_start = None
                        
                        # Update message with remaining time
                        remaining_time = max(0, session.CHALLENGE_TIMEOUT - (current_time - session.challenge_start_time))
                        session.message = f"Please {session.challenge.replace('_', ' ')} - {remaining_time:.0f}s remaining"
                
                # NEW: Return to center phase
                elif session.challenge_met and session.status == "return_center":
                    # Check if user has returned to center position
                    deviation = abs(avg_ratio - session.center_ratio_avg)
                    
                    if deviation < 0.05:  # User is back to center
                        if not hasattr(session, 'center_stable_start'):
                            session.center_stable_start = current_time
                        
                        time_stable = current_time - session.center_stable_start
                        session.message = f"Hold center position... ({time_stable:.1f}s/1.0s)"
                        
                        if time_stable >= 1.0:  # Stable for 1 second
                            session.status = "success"
                            session.message = "Liveness verified successfully!"
                            
                            # NOW DO THE FACE MATCHING HERE
                            user = get_user_by_username(username)
                            if user and user[4]:  # user[4] is the face_encoding column
                                print(f"🔍 Starting face matching for user: {username}")
                                
                                # Use enhanced face comparison
                                match_success, confidence = enhanced_face_compare(frame, user[4], tolerance=0.6)
                                
                                print(f"🎯 Face matching result: Match={match_success}, Confidence={confidence:.3f}")
                                
                                # Try with different tolerances if first attempt fails
                                if not match_success and confidence > 0.4:
                                    print(f"🔄 Retrying with relaxed tolerance (0.7)...")
                                    match_success_relaxed, confidence_relaxed = enhanced_face_compare(frame, user[4], tolerance=0.7)
                                    
                                    if match_success_relaxed:
                                        print(f"✅ Match successful with relaxed tolerance: {confidence_relaxed:.3f}")
                                        match_success, confidence = match_success_relaxed, confidence_relaxed
                                
                                # Success threshold - you can adjust this
                                SUCCESS_THRESHOLD = 0.5
                                
                                if match_success and confidence > SUCCESS_THRESHOLD:
                                    update_last_login(username)
                                    print(f"🎉 Login successful for {username} (confidence: {confidence:.3f})")
                                    return jsonify({
                                        'session_data': session.to_dict(),
                                        'face_match': True,
                                        'confidence': confidence,
                                        'login_success': True,
                                        'ai_status': 'real',
                                        'match_method': 'enhanced'
                                    }), 200
                                else:
                                    session.status = "failed"
                                    session.message = f"Face verification failed (confidence: {confidence:.2f})"
                                    print(f"❌ Face verification failed for {username} (confidence: {confidence:.3f})")
                                    
                                    # Provide helpful feedback
                                    if confidence < 0.3:
                                        session.message += " - Face not recognized"
                                    elif confidence < SUCCESS_THRESHOLD:
                                        session.message += " - Low confidence match"
                                    
                                    return jsonify({
                                        'session_data': session.to_dict(),
                                        'face_match': False,
                                        'confidence': confidence,
                                        'debug_info': f"Threshold: {SUCCESS_THRESHOLD}, Got: {confidence:.3f}"
                                    }), 200
                            else:
                                session.status = "failed"
                                session.message = "No face data registered for this user"
                                print(f"❌ No face data found for user: {username}")
                                return jsonify({
                                    'session_data': session.to_dict(),
                                    'face_match': False,
                                    'confidence': 0.0
                                }), 200
                    else:
                        # Reset stability timer if not centered
                        if hasattr(session, 'center_stable_start'):
                            delattr(session, 'center_stable_start')
                        session.message = "Please look straight ahead for face verification"
        else:
            session.message = "No face detected"
        
        return jsonify({
            'session_data': session.to_dict(),
            'ai_status': 'real' if ai_says_real else 'checking'
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500    
@app.route('/api/user/profile', methods=['GET'])  # Changed route
@token_required
def get_user_info(username):
    try:
        user = get_user_by_username(username)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'username': user[1],
            'email': user[2],
            'registration_date': user[5],
            'last_login': user[6] if user[6] else 'Never'  # Handle null case
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Additional utility route for debugging face matching
@app.route('/api/debug-face-match', methods=['POST'])
@token_required
def debug_face_match(username):
    """
    Debug endpoint to test face matching with detailed output
    """
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        frame = base64_to_image(frame_data)
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        
        user = get_user_by_username(username)
        if not user or not user[4]:
            return jsonify({'error': 'No face data registered'}), 400
        
        # Run enhanced face comparison with debug info
        match_success, confidence = enhanced_face_compare(frame, user[4], tolerance=0.6)
        
        # Test with multiple tolerances
        tolerance_tests = {}
        for tolerance in [0.4, 0.5, 0.6, 0.7, 0.8]:
            test_match, test_conf = enhanced_face_compare(frame, user[4], tolerance=tolerance)
            tolerance_tests[tolerance] = {
                'match': test_match,
                'confidence': test_conf
            }
        
        return jsonify({
            'primary_result': {
                'match': match_success,
                'confidence': confidence
            },
            'tolerance_tests': tolerance_tests,
            'recommendations': [
                f"Current confidence: {confidence:.3f}",
                f"Recommended tolerance: {0.7 if confidence > 0.5 else 0.8}",
                "Try better lighting if confidence is low",
                "Ensure face is clearly visible and centered"
            ]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Migration route (run this once to migrate existing users)
@app.route('/api/migrate-face-data', methods=['POST'])
@token_required
def migrate_face_data_route(username):
    """
    Migrate existing face data to the new enhanced format
    Only run this once after deploying the new system
    """
    try:
        # Check if user is admin or has special permissions
        # You might want to add proper admin authentication here
        
        migrate_existing_face_data()
        
        return jsonify({
            'message': 'Face data migration completed',
            'status': 'success'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/request-document', methods=['POST'])
@token_required
def request_document(username):
    try:
        data = request.get_json()
        document_type = data.get('document_type')
        
        if not document_type:
            return jsonify({'error': 'Document type required'}), 400
        
        doc_names = {
            'birth-certificate': 'Birth Certificate',
            'marriage-certificate': 'Marriage Certificate',
            'identity-card': 'Identity Card',
            'passport': 'Passport',
            'extract-documents': 'Document Extraction',
            'verification-history': 'Verification History'
        }
        
        messages = {
            'birth-certificate': 'Birth Certificate request submitted! Our AI will process your request and you\'ll receive email confirmation shortly.',
            'marriage-certificate': 'Marriage Certificate request submitted! Document processing will begin immediately.',
            'identity-card': 'Identity Card request submitted! Your new ID will be processed within 3-5 business days.',
            'passport': 'Passport application submitted! Processing time is 2-3 weeks for standard service.',
            'extract-documents': 'Document extraction service initiated! Our AI will analyze and digitize your documents.',
            'verification-history': 'Verification history accessed! View your complete biometric authentication log.'
        }
        
        if document_type not in doc_names:
            return jsonify({'error': 'Invalid document type'}), 400
            
        return jsonify({
            'message': messages.get(document_type, f'{doc_names[document_type]} request submitted successfully'),
            'request_id': f'REQ_{int(time.time())}',
            'status': 'pending',
            'document_type': document_type
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add this endpoint to handle document processing
@app.route('/api/process-document', methods=['POST'])
@token_required
def process_document(username):
    try:
        data = request.get_json()
        image_data = data.get('image')
        document_type = data.get('document_type', 'generic')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Improve image quality for better OCR
            image = improve_image_for_ocr(image)
            
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Extract text using OCR with better configuration
        try:
            # Configure Tesseract for better results
            custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:-/\\ '
            extracted_text = pytesseract.image_to_string(image, config=custom_config)
            
            if not extracted_text.strip():
                # Try with different PSM mode
                custom_config = '--oem 3 --psm 11'
                extracted_text = pytesseract.image_to_string(image, config=custom_config)
                
        except Exception as e:
            return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500
        
        # Process extracted text based on document type
        extracted_data = process_extracted_text(extracted_text, document_type)
        
        # Store extraction result
        extraction_record = {
            'username': username,
            'document_type': document_type,
            'extracted_data': extracted_data,
            'raw_text': extracted_text,
            'timestamp': datetime.datetime.now().isoformat(),
            'extraction_id': f'EXT_{int(time.time())}'
        }
        
        return jsonify({
            'success': True,
            'extracted_data': extracted_data,
            'raw_text': extracted_text,  # Include raw text for debugging
            'extraction_id': extraction_record['extraction_id']
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
def extract_generic_data(text, text_upper):
    """Generic extraction for unknown document types"""
    data = {}
    
    # Generic patterns
    name_match = re.search(r'NAME[:\s]+([A-Z\s]+)', text_upper)
    if name_match:
        data['name'] = name_match.group(1).strip()
    
    date_matches = re.findall(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', text)
    if date_matches:
        data['dates_found'] = date_matches
    
    number_matches = re.findall(r'\b[A-Z0-9]{6,}\b', text_upper)
    if number_matches:
        data['numbers_found'] = number_matches
    
    return data
def improve_image_for_ocr(image):
    """Improve image quality for better OCR results"""
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Increase contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Resize if too small (OCR works better on larger images)
        width, height = image.size
        if width < 1000 or height < 1000:
            scale_factor = max(1000/width, 1000/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        print(f"Image enhancement failed: {e}")
        return image

def process_extracted_text(text, document_type):
    """
    Enhanced text processing with better pattern matching and generic fallback
    """
    if not text or not text.strip():
        return {}
    
    text_upper = text.upper()
    extracted_data = {}
    
    # Try specific document type extraction
    if document_type == 'birth-certificate':
        extracted_data = extract_birth_certificate_data(text, text_upper)
    elif document_type == 'marriage-certificate':
        extracted_data = extract_marriage_certificate_data(text, text_upper)
    elif document_type == 'identity-card':
        extracted_data = extract_id_card_data(text, text_upper)
    elif document_type == 'passport':
        extracted_data = extract_passport_data(text, text_upper)
    
    # If no specific extraction worked, try generic patterns
    if not extracted_data or len(extracted_data) == 0:
        extracted_data = extract_generic_data(text, text_upper)
    
    # Enhanced generic extraction as fallback
    if not extracted_data or len(extracted_data) == 0:
        extracted_data = extract_enhanced_generic_data(text, text_upper)
    
    return extracted_data

def extract_enhanced_generic_data(text, text_upper):
    """Enhanced generic extraction with more flexible patterns"""
    data = {}
    
    # More flexible name patterns
    name_patterns = [
        r'NAME[:\s]*([A-Z][A-Za-z\s]{2,30})',
        r'FULL NAME[:\s]*([A-Z][A-Za-z\s]{2,50})',
        r'^([A-Z][A-Za-z]+\s+[A-Z][A-Za-z\s]+)',  # First line names
        r'MR\.?\s+([A-Z][A-Za-z\s]+)',
        r'MS\.?\s+([A-Z][A-Za-z\s]+)',
        r'MRS\.?\s+([A-Z][A-Za-z\s]+)',
    ]
    
    # Find names
    for i, pattern in enumerate(name_patterns):
        matches = re.findall(pattern, text if i > 2 else text_upper)
        if matches:
            data[f'name_{i+1}'] = matches[0].strip()
            if i == 0:  # First match is most likely the main name
                break
    
    # Enhanced date patterns
    date_patterns = [
        r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',
        r'\b(\d{2,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b',
        r'\b(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{2,4})\b',
    ]
    
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text_upper)
        dates_found.extend(matches)
    
    if dates_found:
        data['dates_found'] = list(set(dates_found))  # Remove duplicates
    
    # ID/Reference numbers
    number_patterns = [
        r'\b([A-Z]{2,3}\d{6,12})\b',  # Passport style
        r'\b(\d{8,15})\b',            # Long numbers
        r'\b([A-Z]\d{7,12})\b',       # Letter + numbers
        r'NO[:\s]*([A-Z0-9]{6,})',    # After "NO:"
        r'ID[:\s]*([A-Z0-9]{6,})',    # After "ID:"
    ]
    
    numbers_found = []
    for pattern in number_patterns:
        matches = re.findall(pattern, text_upper)
        numbers_found.extend(matches)
    
    if numbers_found:
        data['numbers_found'] = list(set(numbers_found))
    
    # Addresses (basic pattern)
    address_patterns = [
        r'ADDRESS[:\s]*([A-Z0-9\s,.-]{10,100})',
        r'(\d+\s+[A-Z][A-Za-z\s]{5,50})',
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, text_upper)
        if match:
            data['address'] = match.group(1).strip()
            break
    
    # Phone numbers
    phone_pattern = r'(\+?\d{1,3}[-.\s]?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4})'
    phone_matches = re.findall(phone_pattern, text)
    if phone_matches:
        data['phone_numbers'] = phone_matches
    
    # Email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_pattern, text)
    if email_matches:
        data['email_addresses'] = email_matches
    
    # Extract any words that might be important (filter out common words)
    common_words = {'THE', 'AND', 'OR', 'BUT', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'WITH', 'BY', 'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'MUST', 'CAN', 'CANNOT'}
    
    important_words = []
    words = re.findall(r'\b[A-Z]{3,}\b', text_upper)
    for word in words:
        if word not in common_words and len(word) >= 3:
            important_words.append(word)
    
    if important_words:
        data['important_terms'] = list(set(important_words))[:10]  # Limit to 10
    
    # Add raw text length and line count for reference
    data['text_stats'] = {
        'character_count': len(text),
        'line_count': len(text.split('\n')),
        'word_count': len(text.split())
    }
    
    return data

# Add better error handling for missing dependencies
try:
    import pytesseract
    from PIL import Image, ImageEnhance
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pytesseract pillow")
    
# Make sure Tesseract is installed and accessible
try:
    pytesseract.get_tesseract_version()
except:
    print("Tesseract OCR not found. Please install Tesseract OCR:")
    print("- Ubuntu/Debian: sudo apt install tesseract-ocr")
    print("- macOS: brew install tesseract")
    print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")

# Optional: Add endpoint to get document extraction history
@app.route('/api/document-extractions', methods=['GET'])
@token_required
def get_document_extractions(username):
    try:
        # If you implement database storage, retrieve user's extraction history here
        # For now, return empty list
        return jsonify({
            'extractions': [],
            'message': 'Document extraction history (implement database storage)'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Add this route for document generation (replace the existing request-document route)
@app.route('/api/generate-document', methods=['POST'])
@token_required
def generate_document(username):
    try:
        data = request.get_json()
        doc_type = data.get('document_type')
        form_data = data.get('form_data', {})
        
        if not doc_type:
            return jsonify({'error': 'Document type required'}), 400
        
        if not form_data:
            return jsonify({'error': 'Form data required'}), 400
        
        # Get user information for the document
        user = get_user_by_username(username)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Generate PDF based on document type
        if doc_type == 'birth-certificate':
            pdf_buffer = generate_birth_certificate_pdf(form_data, username)
        elif doc_type == 'marriage-certificate':
            pdf_buffer = generate_marriage_certificate_pdf(form_data, username)
        elif doc_type == 'identity-card':
            # Get user's face photo - try form data first, then database
            user_face = form_data.get('photoData') or get_user_face_photo(username)
            pdf_buffer = generate_identity_card_pdf(form_data, username, user_face)
        elif doc_type == 'passport':
            user_face = form_data.get('photoData') or get_user_face_photo(username)
            pdf_buffer = generate_passport_pdf(form_data, username, user_face)
        else:
            return jsonify({'error': 'Invalid document type'}), 400
        
        # Convert PDF buffer to base64 for JSON response
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
        
        # Create download filename
        filename = f"{doc_type.replace('-', '_')}_{username}_{int(time.time())}.pdf"
        
        return jsonify({
            'success': True,
            'pdf_data': pdf_base64,
            'filename': filename,
            'document_type': doc_type,
            'generated_at': datetime.datetime.now().isoformat(),
            'message': f'{doc_type.replace("-", " ").title()} generated successfully'
        }), 200
        
    except Exception as e:
        print(f"Document generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Document generation failed: {str(e)}'}), 500


def get_user_face_photo(username):
    """Get user's registered face photo from database"""
    try:
        user = get_user_by_username(username)
        if user and user[4]:  # user[4] is the face_encoding column
            face_data = json.loads(user[4])
            if 'image_path' in face_data and os.path.exists(face_data['image_path']):
                # Read the image file and convert to base64
                with open(face_data['image_path'], 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    return f"data:image/jpeg;base64,{img_base64}"
        return None
    except Exception as e:
        print(f"Error getting user face photo: {e}")
        return None

def generate_birth_certificate_pdf(data, username):
    """Generate birth certificate PDF"""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header with official styling
    p.setFont("Helvetica-Bold", 28)
    p.drawCentredString(width/2, height-80, "REPUBLIC OF SECUREID")
    
    p.setFont("Helvetica-Bold", 24)
    p.drawCentredString(width/2, height-120, "CERTIFICATE OF BIRTH")
    
    p.setFont("Helvetica", 12)
    p.drawCentredString(width/2, height-150, "Official Government Document")
    
    # Add certificate number
    cert_number = f"BC-{int(time.time())}-{username.upper()[:3]}"
    p.setFont("Helvetica-Bold", 10)
    p.drawString(450, height-40, f"Certificate No: {cert_number}")
    
    # Content fields
    y_position = height - 220
    fields = [
        ("Full Name:", f"{data.get('firstName', '')} {data.get('middleName', '')} {data.get('lastName', '')}".strip()),
        ("Date of Birth:", data.get('birthDate', '')),
        ("Place of Birth:", data.get('birthPlace', '')),
        ("Gender:", data.get('gender', '')),
        ("Father's Full Name:", data.get('fatherName', '')),
        ("Mother's Full Name:", data.get('motherName', '')),
        ("Registration Number:", data.get('registrationNumber', cert_number)),
        ("Nationality:", data.get('nationality', 'SecureID Citizen'))
    ]
    
    for label, value in fields:
        p.setFont("Helvetica-Bold", 12)
        p.drawString(80, y_position, label)
        p.setFont("Helvetica", 12)
        p.drawString(250, y_position, str(value))
        # Add underline
        p.line(250, y_position-5, 500, y_position-5)
        y_position -= 35
    
    # Official stamps and signatures area
    p.setFont("Helvetica-Bold", 10)
    p.drawString(80, 150, "Registrar's Signature:")
    p.line(200, 145, 350, 145)
    
    p.drawString(380, 150, "Official Seal:")
    p.circle(450, 130, 30, fill=0)
    p.drawCentredString(450, 130, "OFFICIAL\nSEAL")
    
    # Footer
    p.setFont("Helvetica", 10)
    p.drawCentredString(width/2, 80, f"Issued by SecureID Biometric Registration System")
    p.drawCentredString(width/2, 65, f"Date of Issue: {datetime.datetime.now().strftime('%B %d, %Y')}")
    p.drawCentredString(width/2, 50, f"Verified by AI-Powered Identity System")
    
    # Add security watermark
    p.setFont("Helvetica", 48)
    p.setFillGray(0.9)
    p.saveState()
    p.translate(width/2, height/2)
    p.rotate(45)
    p.drawCentredString(0, 0, "SECUREID")
    p.restoreState()
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

def generate_marriage_certificate_pdf(data, username):
    """Generate marriage certificate PDF"""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header
    p.setFont("Helvetica-Bold", 28)
    p.drawCentredString(width/2, height-80, "REPUBLIC OF SECUREID")
    
    p.setFont("Helvetica-Bold", 24)
    p.drawCentredString(width/2, height-120, "CERTIFICATE OF MARRIAGE")
    
    p.setFont("Helvetica", 12)
    p.drawCentredString(width/2, height-150, "Official Government Document")
    
    # Certificate number
    cert_number = f"MC-{int(time.time())}-{username.upper()[:3]}"
    p.setFont("Helvetica-Bold", 10)
    p.drawString(450, height-40, f"Certificate No: {cert_number}")
    
    # Marriage details
    y_position = height - 200
    p.setFont("Helvetica", 12)
    p.drawCentredString(width/2, y_position, "This is to certify that")
    
    y_position -= 40
    
    # Spouse 1 details
    p.setFont("Helvetica-Bold", 14)
    p.drawCentredString(width/2, y_position, f"{data.get('spouse1Name', '')}")
    y_position -= 25
    p.setFont("Helvetica", 10)
    p.drawCentredString(width/2, y_position, f"Son/Daughter of {data.get('spouse1Parent', '')}")
    
    y_position -= 40
    p.setFont("Helvetica", 12)
    p.drawCentredString(width/2, y_position, "and")
    
    y_position -= 40
    
    # Spouse 2 details
    p.setFont("Helvetica-Bold", 14)
    p.drawCentredString(width/2, y_position, f"{data.get('spouse2Name', '')}")
    y_position -= 25
    p.setFont("Helvetica", 10)
    p.drawCentredString(width/2, y_position, f"Son/Daughter of {data.get('spouse2Parent', '')}")
    
    y_position -= 40
    p.setFont("Helvetica", 12)
    p.drawCentredString(width/2, y_position, f"were lawfully married on {data.get('marriageDate', '')}")
    p.drawCentredString(width/2, y_position-20, f"at {data.get('marriagePlace', '')}")
    
    # Additional fields
    y_position -= 80
    fields = [
        ("Marriage Registration Number:", cert_number),
        ("Witnesses:", data.get('witnesses', 'As per records')),
        ("Officiant:", data.get('officiant', 'SecureID Authorized Official'))
    ]
    
    for label, value in fields:
        p.setFont("Helvetica-Bold", 10)
        p.drawString(80, y_position, label)
        p.setFont("Helvetica", 10)
        p.drawString(280, y_position, str(value))
        y_position -= 25
    
    # Signatures
    p.setFont("Helvetica-Bold", 10)
    p.drawString(80, 150, "Registrar's Signature:")
    p.line(200, 145, 350, 145)
    
    p.drawString(380, 150, "Official Seal:")
    p.circle(450, 130, 30, fill=0)
    p.drawCentredString(450, 130, "OFFICIAL\nSEAL")
    
    # Footer
    p.setFont("Helvetica", 10)
    p.drawCentredString(width/2, 80, f"Issued by SecureID Marriage Registration Office")
    p.drawCentredString(width/2, 65, f"Date of Issue: {datetime.datetime.now().strftime('%B %d, %Y')}")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

def generate_identity_card_pdf(data, username, user_face_data):
    """Generate identity card PDF (front and back) with proper photo handling"""
    buffer = BytesIO()
    # ID Card dimensions (3.375" x 2.125")
    card_width, card_height = 243, 153  # in points
    
    p = canvas.Canvas(buffer, pagesize=(card_width * 2 + 20, card_height + 40))
    
    # Front of card
    x_offset = 10
    y_offset = 20
    
    # Draw card border
    p.rect(x_offset, y_offset, card_width, card_height, stroke=1, fill=0)
    
    # Header
    p.setFont("Helvetica-Bold", 8)
    p.drawCentredString(x_offset + card_width/2, y_offset + card_height - 15, "REPUBLIC OF SECUREID")
    p.setFont("Helvetica-Bold", 6)
    p.drawCentredString(x_offset + card_width/2, y_offset + card_height - 25, "NATIONAL IDENTITY CARD")
    
    # Photo area
    photo_x, photo_y = x_offset + 10, y_offset + card_height - 80
    photo_width, photo_height = 45, 55
    
    # Enhanced photo handling
    photo_added = False
    if user_face_data:
        try:
            print(f"📸 Processing photo data for user: {username}")
            
            # Handle different photo data formats
            img_data = None
            if isinstance(user_face_data, str):
                if user_face_data.startswith('data:image'):
                    # Base64 data URL format
                    img_data = base64.b64decode(user_face_data.split(',')[1])
                else:
                    # Direct base64 string
                    img_data = base64.b64decode(user_face_data)
            
            if img_data:
                # Create PIL image to verify and potentially resize
                from PIL import Image
                pil_image = Image.open(BytesIO(img_data))
                
                # Resize if necessary (maintain aspect ratio)
                max_size = (200, 240)  # Higher resolution for better quality
                pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert back to bytes
                img_buffer = BytesIO()
                pil_image.save(img_buffer, format='JPEG', quality=95)
                img_buffer.seek(0)
                
                # Create ImageReader object
                img = ImageReader(img_buffer)
                p.drawImage(img, photo_x, photo_y, width=photo_width, height=photo_height)
                photo_added = True
                print(f"✅ Photo successfully added to PDF")
                
        except Exception as e:
            print(f"❌ Error processing photo: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback if photo couldn't be added
    if not photo_added:
        print("📷 Using fallback photo placeholder")
        p.rect(photo_x, photo_y, photo_width, photo_height, stroke=1, fill=1)
        p.setFont("Helvetica", 6)
        p.drawCentredString(photo_x + photo_width/2, photo_y + photo_height/2, "PHOTO")
    
    # Personal information
    info_x = photo_x + photo_width + 10
    info_y = y_offset + card_height - 40
    
    p.setFont("Helvetica-Bold", 6)
    p.drawString(info_x, info_y, "NAME:")
    p.setFont("Helvetica", 6)
    p.drawString(info_x, info_y - 8, f"{data.get('firstName', '')} {data.get('lastName', '')}")
    
    p.setFont("Helvetica-Bold", 6)
    p.drawString(info_x, info_y - 20, "ID NUMBER:")
    id_number = data.get('idNumber', f"ID{int(time.time())}{username[:2].upper()}")
    p.setFont("Helvetica", 6)
    p.drawString(info_x, info_y - 28, id_number)
    
    p.setFont("Helvetica-Bold", 6)
    p.drawString(info_x, info_y - 40, "DATE OF BIRTH:")
    p.setFont("Helvetica", 6)
    dob_formatted = data.get('dateOfBirth', data.get('dob', ''))
    if dob_formatted:
        try:
            dob_formatted = datetime.datetime.strptime(dob_formatted, '%Y-%m-%d').strftime('%m/%d/%Y')
        except:
            pass  # Keep original format if parsing fails
    p.drawString(info_x, info_y - 48, dob_formatted)
    
    p.setFont("Helvetica-Bold", 6)
    p.drawString(info_x, info_y - 60, "NATIONALITY:")
    p.setFont("Helvetica", 6)
    p.drawString(info_x, info_y - 68, data.get('nationality', 'SECUREID'))
    
    # Expiry date
    expiry_date = (datetime.datetime.now() + datetime.timedelta(days=3650)).strftime('%m/%d/%Y')  # 10 years
    p.setFont("Helvetica-Bold", 5)
    p.drawString(x_offset + 10, y_offset + 15, f"EXPIRES: {expiry_date}")
    
    # Back of card
    back_x_offset = x_offset + card_width + 10
    
    # Draw back card border
    p.rect(back_x_offset, y_offset, card_width, card_height, stroke=1, fill=0)
    
    # Back content
    p.setFont("Helvetica-Bold", 6)
    p.drawString(back_x_offset + 10, y_offset + card_height - 20, "ADDRESS:")
    p.setFont("Helvetica", 5)
    address = data.get('address', '')
    address_lines = address.replace(', ', '\n').split('\n')
    for i, line in enumerate(address_lines[:3]):  # Max 3 lines
        p.drawString(back_x_offset + 10, y_offset + card_height - 30 - (i * 8), line.strip())
    
    p.setFont("Helvetica-Bold", 6)
    p.drawString(back_x_offset + 10, y_offset + card_height - 70, "EMERGENCY CONTACT:")
    p.setFont("Helvetica", 5)
    p.drawString(back_x_offset + 10, y_offset + card_height - 80, data.get('emergencyContact', 'N/A'))
    
    # Signature strip
    p.setFont("Helvetica-Bold", 5)
    p.drawString(back_x_offset + 10, y_offset + 40, "SIGNATURE:")
    p.line(back_x_offset + 50, y_offset + 35, back_x_offset + 150, y_offset + 35)
    
    # Security features
    p.setFont("Helvetica", 4)
    p.drawString(back_x_offset + 10, y_offset + 10, f"Issued: {datetime.datetime.now().strftime('%m/%d/%Y')}")
    p.drawString(back_x_offset + 100, y_offset + 10, f"AI Verified: ✓")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

def generate_passport_pdf(data, username, user_face_data):
    """Generate passport-style document PDF"""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header
    p.setFont("Helvetica-Bold", 24)
    p.drawCentredString(width/2, height-60, "REPUBLIC OF SECUREID")
    
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(width/2, height-90, "PASSPORT")
    
    # Passport number
    passport_number = f"P{int(time.time())}{username[:3].upper()}"
    p.setFont("Helvetica-Bold", 12)
    p.drawString(450, height-40, f"Passport No: {passport_number}")
    
    # Photo area
    photo_x, photo_y = 80, height - 250
    photo_width, photo_height = 100, 120
    
    if user_face_data:
        try:
            if ',' in user_face_data:
                img_data = base64.b64decode(user_face_data.split(',')[1])
            else:
                img_data = base64.b64decode(user_face_data)
            
            img = ImageReader(BytesIO(img_data))
            p.drawImage(img, photo_x, photo_y, width=photo_width, height=photo_height)
        except:
            p.rect(photo_x, photo_y, photo_width, photo_height, stroke=1, fill=1)
            p.setFont("Helvetica", 8)
            p.drawCentredString(photo_x + photo_width/2, photo_y + photo_height/2, "PASSPORT\nPHOTO")
    else:
        p.rect(photo_x, photo_y, photo_width, photo_height, stroke=1, fill=1)
        p.setFont("Helvetica", 8)
        p.drawCentredString(photo_x + photo_width/2, photo_y + photo_height/2, "PASSPORT\nPHOTO")
    
    # Personal information
    info_x = 220
    info_y = height - 140
    
    fields = [
        ("Type/Type:", "P"),
        ("Country Code/Code pays:", "SEC"),
        ("Passport No./No du passeport:", passport_number),
        ("Surname/Nom:", data.get('lastName', '').upper()),
        ("Given Names/Prénoms:", data.get('firstName', '').upper()),
        ("Nationality/Nationalité:", data.get('nationality', 'SECUREID')),
        ("Date of Birth/Date de naissance:", data.get('dateOfBirth', '')),
        ("Sex/Sexe:", data.get('gender', '').upper()[:1]),
        ("Place of Birth/Lieu de naissance:", data.get('placeOfBirth', '')),
        ("Date of Issue/Date de délivrance:", datetime.datetime.now().strftime('%d %b %Y')),
        ("Date of Expiry/Date d'expiration:", (datetime.datetime.now() + datetime.timedelta(days=3650)).strftime('%d %b %Y')),
        ("Authority/Autorité:", "SecureID Passport Office")
    ]
    
    for label, value in fields:
        p.setFont("Helvetica-Bold", 8)
        p.drawString(info_x, info_y, label)
        p.setFont("Helvetica", 10)
        p.drawString(info_x, info_y - 12, str(value))
        info_y -= 30
    
    # Machine readable zone (MRZ)
    p.setFont("Courier", 10)
    mrz_y = 120
    # Line 1
    mrz_line1 = f"P<SEC{data.get('lastName', '')[:14].ljust(14).upper()}<<{data.get('firstName', '')[:14].ljust(14).upper()}<<<"
    p.drawString(80, mrz_y, mrz_line1[:44])
    
    # Line 2
    birth_date = data.get('dateOfBirth', '01/01/1990').replace('/', '')[-6:]  # YYMMDD
    expiry_date = (datetime.datetime.now() + datetime.timedelta(days=3650)).strftime('%y%m%d')
    mrz_line2 = f"{passport_number.ljust(9)[:9]}SEC{birth_date}M{expiry_date}<<<<<<<<<<<<<<<7"
    p.drawString(80, mrz_y - 15, mrz_line2[:44])
    
    # Security note
    p.setFont("Helvetica", 8)
    p.drawString(80, 50, "This document contains security features and is protected by international law.")
    p.drawString(80, 35, f"AI-Verified Identity • Biometrically Secured • Issued: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer
# Add this endpoint to your Flask backend (paste.txt)

@app.route('/api/user/face-photo', methods=['GET'])
@token_required
def get_user_face_photo_endpoint(username):
    """
    Get the user's registered face photo
    """
    try:
        print(f"🔍 Getting face photo for user: {username}")
        
        user = get_user_by_username(username)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if not user[4]:  # user[4] is the face_encoding column
            return jsonify({
                'face_photo': None,
                'message': 'No face photo registered'
            }), 200
        
        try:
            face_data = json.loads(user[4])
            
            # Check if we have an image path
            if 'image_path' in face_data and os.path.exists(face_data['image_path']):
                print(f"📂 Found face image at: {face_data['image_path']}")
                
                # Read the image file and convert to base64
                with open(face_data['image_path'], 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    face_photo_data = f"data:image/jpeg;base64,{img_base64}"
                    
                    print(f"✅ Successfully retrieved face photo (size: {len(img_data)} bytes)")
                    
                    return jsonify({
                        'face_photo': face_photo_data,
                        'message': 'Face photo retrieved successfully',
                        'timestamp': face_data.get('timestamp', 'Unknown')
                    }), 200
            
            else:
                print("❌ No valid image path found in face data")
                return jsonify({
                    'face_photo': None,
                    'message': 'Face photo file not found'
                }), 404
                
        except json.JSONDecodeError:
            print("❌ Invalid face data format in database")
            return jsonify({
                'face_photo': None,
                'message': 'Invalid face data format'
            }), 500
            
    except Exception as e:
        print(f"❌ Error retrieving face photo: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to retrieve face photo: {str(e)}'}), 500
@app.route('/document-generator.html')
def document_generator():
    return render_template('document-generator.html')
if __name__ == '__main__':
    print("🚀 Starting SecureID Backend Server...")
    print("📊 Features enabled:")
    print("  ✅ User Registration & Authentication")
    print("  ✅ Face Capture & Storage")
    print("  ✅ Liveness Detection")
    print("  ✅ Anti-Spoofing AI")
    print("  ✅ Face Matching")
    print("  ✅ Document Request Services")
    print("\n🌐 Server starting on http://localhost:5004")
    
    app.run(debug=True, host='0.0.0.0', port=5004)