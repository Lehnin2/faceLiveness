# Identity Recognition & Document Processing System

A Flask-based application that combines facial recognition, deep learning, OCR capabilities, and document generation for identity verification and processing.

## Features

- **Facial Recognition**: Real-time face detection and recognition using MediaPipe and face_recognition library
- **Deep Learning**: ResNet-based model for advanced image classification
- **Optical Character Recognition (OCR)**: Extract text from documents using Tesseract
- **User Authentication**: JWT-based authentication system with secure password hashing
- **Document Generation**: Automatic PDF document creation with ReportLab
- **Web Interface**: Interactive Flask web application with CORS support
- **Database**: SQLite database for user management and face encoding storage

## Project Structure

```
.
├── app.py                          # Main Flask application
├── clear_users.py                  # Utility to reset user database
├── debug.py                        # Debugging utilities
├── kaka.py                         # Additional utilities/scripts
├── requirements.txt                # Python dependencies
├── models/
│   └── resnet_checkpoint_epoch_8.pt  # Pre-trained ResNet model checkpoint
├── static/
│   └── faces/                      # Directory for storing processed face images
├── templates/
│   ├── index.html                  # Main web interface
│   └── document-generator.html     # Document generation interface
└── README.md                       # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR engine (for Windows: [Download from GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki))
- Virtual environment (recommended)

### Setup Steps

1. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract** (if using OCR features):
   - **Windows**: Download installer from [tesseract-ocr](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. **Configure Tesseract path** (if needed):
   Edit `app.py` and uncomment/update the Tesseract path for your system.

## Usage

### Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000` (or the configured port).

### Key Modules

- **app.py**: Main Flask application with API endpoints
- **clear_users.py**: Reset user database (remove all users)
- **debug.py**: Testing and debugging utilities
- **kaka.py**: Additional helper functions

## Dependencies

- **Flask**: Web framework
- **Flask-CORS**: Cross-Origin Resource Sharing support
- **OpenCV**: Computer vision library for image processing
- **MediaPipe**: Face detection and mesh
- **PyTorch & Torchvision**: Deep learning framework and models
- **Pillow**: Image processing
- **PyJWT**: JSON Web Token authentication
- **face_recognition**: High-level face recognition library
- **pytesseract**: OCR text extraction
- **reportlab**: PDF document generation
- **NumPy**: Numerical computing

## API Endpoints

Available endpoints for user authentication, face recognition, document generation, and more (details in app.py).

## Database

The application uses SQLite (`users.db`) with the following tables:
- **users**: Stores user credentials, email, password hash, and face encodings

## Configuration

Update the following in `app.py` before deployment:
- `SECRET_KEY`: Change from default value to a secure random key
- Tesseract path: Configure for your system if using OCR
- CORS settings: Adjust for production environment

## Security Notes

⚠️ **Development Only**: This project uses hardcoded secrets. For production:
- Use environment variables for sensitive data
- Implement proper HTTPS
- Add rate limiting
- Use secure session management

## Development & Debugging

- Use `debug.py` for testing individual components
- Use `clear_users.py` to reset the database during development
- Enable Flask debug mode for development (not recommended for production)

## License

[Add your license here]

## Contributors

[Add contributors information here]

## Support

For issues, questions, or contributions, please contact the development team.
