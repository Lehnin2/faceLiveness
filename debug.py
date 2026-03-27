# Debug script to test your current face matching system
# Run this to see what's happening with your existing data

import sqlite3
import json
import numpy as np
import cv2
import base64
import face_recognition
from PIL import Image
import io

def debug_existing_user_data():
    """Debug existing users in your database"""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT username, face_encoding FROM users WHERE face_encoding IS NOT NULL")
        users = cursor.fetchall()
        
        print(f"🔍 Found {len(users)} users with face data")
        
        for username, face_data in users:
            print("\n" + "="*50)
            print(f"👤 Testing user: {username}")
            print("="*50)
            
            if not face_data:
                print("❌ No face data")
                continue
            
            # Check what type of data we have
            print(f"📊 Face data length: {len(face_data)} characters")
            print(f"📝 First 100 chars: {face_data[:100]}")
            
            if face_data.startswith('data:image'):
                print("📷 Data type: Base64 image")
                test_base64_image_data(face_data)
            elif face_data.startswith('[') or face_data.startswith('{'):
                print("🧬 Data type: JSON encoding")
                test_json_encoding_data(face_data)
            else:
                print("❓ Unknown data type")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

def test_base64_image_data(base64_data):
    """Test base64 image data"""
    try:
        # Extract base64 part
        if ',' in base64_data:
            header, data = base64_data.split(',', 1)
            print(f"🏷️  Header: {header}")
        else:
            data = base64_data
        
        # Decode
        image_bytes = base64.b64decode(data)
        print(f"📏 Image bytes length: {len(image_bytes)}")
        
        # Try to load as image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("❌ Failed to decode as image")
            return False
        
        print(f"✅ Image decoded successfully: {image.shape}")
        
        # Try face detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        print(f"👥 Faces detected: {len(face_locations)}")
        
        if len(face_locations) > 0:
            encodings = face_recognition.face_encodings(rgb_image, face_locations)
            print(f"🧬 Encodings extracted: {len(encodings)}")
            if len(encodings) > 0:
                print(f"✅ First encoding shape: {encodings[0].shape}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error testing base64 data: {e}")
        return False

def test_json_encoding_data(json_data):
    """Test JSON encoding data"""
    try:
        # Parse JSON
        encoding_list = json.loads(json_data)
        encoding = np.array(encoding_list)
        print(f"✅ JSON parsed successfully: {encoding.shape}")
        print(f"📊 Encoding stats - Min: {encoding.min():.3f}, Max: {encoding.max():.3f}")
        
        # Validate encoding format
        if encoding.shape == (128,):
            print("✅ Encoding shape is correct (128 dimensions)")
            return True
        else:
            print(f"❌ Wrong encoding shape, expected (128,), got {encoding.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing JSON data: {e}")
        return False

def test_face_comparison_with_sample_images():
    """Test face comparison with sample images"""
    print("\n🧪 TESTING FACE COMPARISON")
    print("="*50)
    
    # You can test with two images of the same person
    # For now, let's test with a simple comparison
    
    try:
        # Create two identical test encodings (this should give 100% match)
        test_encoding = np.random.random(128)
        
        # Test identical encodings
        matches = face_recognition.compare_faces([test_encoding], test_encoding, tolerance=0.6)
        distances = face_recognition.face_distance([test_encoding], test_encoding)
        confidence = 1 - distances[0]
        
        print(f"🧪 Identical encoding test:")
        print(f"   Match: {matches[0]}")
        print(f"   Distance: {distances[0]:.6f}")
        print(f"   Confidence: {confidence:.6f}")
        
        # Test slightly different encodings
        test_encoding2 = test_encoding + np.random.normal(0, 0.01, 128)  # Add small noise
        
        matches2 = face_recognition.compare_faces([test_encoding], test_encoding2, tolerance=0.6)
        distances2 = face_recognition.face_distance([test_encoding], test_encoding2)
        confidence2 = 1 - distances2[0]
        
        print(f"🧪 Slightly different encoding test:")
        print(f"   Match: {matches2[0]}")
        print(f"   Distance: {distances2[0]:.6f}")
        print(f"   Confidence: {confidence2:.6f}")
        
    except Exception as e:
        print(f"❌ Error in comparison test: {e}")

def check_opencv_and_face_recognition():
    """Check if OpenCV and face_recognition are working properly"""
    print("\n🔧 CHECKING LIBRARIES")
    print("="*50)
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        import face_recognition
        print(f"✅ face_recognition imported successfully")
        
        # Test with a simple image creation
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        print(f"✅ Color conversion test passed")
        
        # Test face_recognition functions
        locations = face_recognition.face_locations(rgb_image)
        print(f"✅ face_locations function works (found {len(locations)} faces in empty image)")
        
    except Exception as e:
        print(f"❌ Library check failed: {e}")

def suggest_fixes():
    """Suggest potential fixes based on common issues"""
    print("\n💡 POTENTIAL FIXES TO TRY:")
    print("="*50)
    print("1. 🔄 Store both image file AND encoding for redundancy")
    print("2. 📏 Use lower tolerance (0.4-0.5) for initial testing")
    print("3. 🎯 Ensure consistent image preprocessing")
    print("4. 🔍 Add more debugging to see exact values")
    print("5. 📷 Capture face images with better lighting/quality")
    print("6. 🔧 Try different face_recognition models (hog vs cnn)")
    print("7. 📊 Compare face encodings of the same person captured at different times")

if __name__ == "__main__":
    print("🚀 FACE MATCHING DEBUG SCRIPT")
    print("="*50)
    
    check_opencv_and_face_recognition()
    debug_existing_user_data()
    test_face_comparison_with_sample_images()
    suggest_fixes()
    
    print("\n✅ Debug complete!")