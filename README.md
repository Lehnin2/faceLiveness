# Face Liveness Detection

A real-time face liveness detection and anti-spoofing system that combines facial landmark tracking with AI-based detection to verify that a user is a real person and not a photo, video, or mask.

## Features

- **Real-time Face Detection**: Uses MediaPipe for robust facial landmark detection
- **AI Anti-Spoofing Model**: ResNet18-based deep learning model to detect spoofed faces
- **Liveness Challenges**: Three interactive challenges to verify user authenticity:
  - **Blink Detection**: Detect when the user blinks their eyes
  - **Look Left**: Detect gaze direction to the left
  - **Look Right**: Detect gaze direction to the right
- **Gaze Tracking**: Calculates gaze direction using eye landmarks
- **Calibration Phase**: Automatically calibrates the system during initialization
- **GPU Support**: Leverages CUDA when available for faster inference

## Requirements

- Python 3.8+
- Webcam/camera device
- GPU (optional, but recommended for better performance)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd faceLiveness
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the pre-trained model**
   - Ensure `resnet_checkpoint_epoch_8.pt` is in the project root directory
   - If you don't have it, the application will fail to load the anti-spoofing model

## Usage

Run the application:
```bash
python app.py
```

### How It Works

1. **Calibration Phase**: The application starts by calibrating your gaze (30 frames)
   - Keep your face centered and look straight ahead during calibration
   
2. **Challenge Phase**: A random challenge appears on screen
   - Follow the instruction (blink, look left, or look right)
   - The system will verify both your liveness and the challenge completion
   - AI checks run periodically to detect spoofing attempts

3. **Verification**: Once the challenge is completed successfully:
   - The system displays "✅ VERIFIED REAL USER"
   - The application closes automatically

### Exit Conditions

- **Success**: User completes the liveness challenge (displays success for 2 seconds)
- **Timeout**: If challenge is not completed within 10 seconds
- **Spoofing Detected**: If the AI model detects a spoofed face

## Requirements File

See `requirements.txt` for all Python dependencies and versions.

### Key Dependencies

- **opencv-python**: Computer vision library for video capture and processing
- **mediapipe**: Google's framework for multimodal machine learning (face detection)
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities for PyTorch
- **pillow**: Image processing library

## Project Structure

```
faceLiveness/
├── app.py                              # Main application script
├── resnet_checkpoint_epoch_8.pt        # Pre-trained anti-spoofing model
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── venv/                               # Virtual environment (not version controlled)
```

## Configuration

Key parameters in `app.py` that can be tuned:

```python
CALIBRATION_FRAMES = 30      # Frames used for calibration
CHALLENGE_TIMEOUT = 10       # Seconds before challenge times out
POST_SUCCESS_DISPLAY = 2     # Seconds to display success message
ai_check_interval = 0.5      # Seconds between AI checks
AI_BUFFER_SIZE = 5           # Number of AI results for averaging
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Webcam not detected | Check camera permissions and ensure no other app is using the webcam |
| Model fails to load | Verify `resnet_checkpoint_epoch_8.pt` exists in the project root |
| Challenges not detecting | Ensure lighting is adequate and your face is clearly visible |
| Challenge timeouts | Move closer to the camera and perform the action more deliberately |
| Slow performance | Enable GPU support or reduce frame processing rate |

## Model Details

- **Architecture**: ResNet18
- **Input Size**: 224×224 pixels
- **Output Classes**: 2 (Real/Spoof)
- **Normalization**: ImageNet normalization statistics

## Performance

- **GPU (CUDA)**: ~30-50 FPS
- **CPU**: ~5-15 FPS (depending on hardware)

## License

[Add your license here]

## Authors

[Your name/team]

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Disclaimer

This system is designed for demonstration and educational purposes. For production use in security-critical applications, additional validation and testing is recommended.
