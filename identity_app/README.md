# Identity Verification Application

A secure identity verification platform built with Flask that uses advanced facial recognition and liveness detection to verify user identities. The application combines MediaPipe face mesh analysis with an anti-spoofing AI model to ensure that verification attempts are genuine.

## Features

- **User Authentication**: Secure login and registration system with password hashing
- **Liveness Detection**: Real-time facial analysis to detect live faces and prevent spoofing attacks
- **Multi-Level Verification**: Progressive verification levels (0=unverified, 1=basic, 2=full)
- **Face Anti-Spoofing**: AI-powered ResNet model to detect presentation attacks
- **WebSocket Support**: Real-time communication using Socket.IO for live feedback
- **Responsive UI**: Modern web interface built with HTML5, CSS, and JavaScript
- **Database Management**: SQLAlchemy ORM with Flask-Migrate for schema management
- **HTTPS Support**: Self-signed certificate support for secure local development

## Tech Stack

- **Backend**: Flask, Flask-SQLAlchemy, Flask-Login, Flask-SocketIO
- **Computer Vision**: OpenCV, MediaPipe
- **AI/ML**: PyTorch (ResNet-based anti-spoofing model)
- **Database**: SQLite (default), configurable for production databases
- **Frontend**: HTML5, CSS, JavaScript
- **Development**: Flask-Migrate for database migrations

## Project Structure

```
identity_app/
├── run.py                      # Application entry point
├── app/
│   ├── __init__.py            # App factory and extension initialization
│   ├── config.py              # Configuration for different environments
│   ├── auth/                  # Authentication routes and forms
│   │   ├── routes.py          # Login/registration endpoints
│   │   └── forms.py           # WTForms for authentication
│   ├── api/                   # API endpoints (extensible)
│   ├── liveness/              # Liveness detection module
│   │   ├── detector.py        # Core liveness detection logic
│   │   ├── ai_model.py        # Anti-spoofing AI model wrapper
│   │   └── routes.py          # Liveness verification endpoints
│   ├── models/                # Database models
│   │   ├── user.py            # User model
│   │   └── verification.py    # Verification records model
│   ├── static/                # Static assets
│   │   ├── css/               # Stylesheets
│   │   ├── js/                # Client-side scripts
│   │   └── images/            # Image assets
│   └── templates/             # Jinja2 HTML templates
│       ├── base.html          # Base template
│       ├── auth/              # Auth templates
│       ├── dashboard/         # Dashboard templates
│       ├── errors/            # Error page templates
│       └── liveness/          # Liveness verification templates
├── ai_models/                 # Pre-trained AI models
│   ├── resnet_checkpoint_epoch_8.pt  # Anti-spoofing model
│   └── model_utils.py         # Model loading utilities
├── migrations/                # Database migration files (Alembic)
├── tests/                     # Unit and integration tests
└── instance/                  # Instance-specific files (ignored in git)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd identity_app
   ```

2. **Create and activate virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   ```bash
   python run.py init_db
   ```

5. **Create admin user (optional)**
   ```bash
   python run.py create_admin
   ```

6. **Run the application**
   ```bash
   python run.py
   ```

   The application will start with HTTPS at `https://localhost:5005`

   **Note**: You'll see a security warning for the self-signed certificate. Accept it to proceed. This is normal for local development.

## Configuration

Configuration is managed in [app/config.py](app/config.py). Three environments are provided:

- **Development**: Debug mode enabled, HTTP cookies allowed
- **Production**: Debug disabled, HTTPS enforced, secure cookies required
- **Testing**: In-memory SQLite database, CSRF disabled

Set the Flask configuration using the `FLASK_CONFIG` environment variable:

```bash
# Linux/macOS
export FLASK_CONFIG=production

# Windows PowerShell
$env:FLASK_CONFIG="production"

# Windows Command Prompt
set FLASK_CONFIG=production
```

### Key Settings

- `SECRET_KEY`: Change this in production (set via environment variable)
- `DATABASE_URL`: Database connection string (defaults to SQLite)
- `AI_MODEL_PATH`: Path to the anti-spoofing model checkpoint
- `LIVENESS_TIMEOUT`: Maximum time allowed for liveness verification (10 seconds)
- `CALIBRATION_FRAMES`: Number of frames for face calibration (30 frames)

## Usage

### User Registration

1. Navigate to the registration page
2. Create an account with username, email, and password
3. Log in with your credentials

### Identity Verification

1. Access the liveness verification page from your dashboard
2. Allow camera access when prompted
3. Follow the on-screen challenges:
   - **Blink**: Blink naturally when instructed
   - **Look Left**: Turn your head to the left
   - **Look Right**: Turn your head to the right
4. The system will verify your liveness and anti-spoofing detection
5. Verification results are stored in your profile

### Admin Commands

```bash
# Initialize database tables
python run.py init_db

# Create default admin user
python run.py create_admin

# Launch Flask shell for database queries
python run.py shell
```

## Development

### Running Tests

```bash
pytest tests/
```

### Database Migrations

Create a migration after model changes:

```bash
flask db migrate -m "Description of changes"
flask db upgrade
```

### Code Organization

- **Models** (app/models/): Database schema definitions
- **Routes** (app/*/routes.py): HTTP endpoint handlers
- **Forms** (app/*/forms.py): Request validation and forms
- **Templates** (app/templates/): HTML rendering
- **Static** (app/static/): CSS, JavaScript, images

## API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `GET /auth/logout` - User logout

### Liveness Verification
- `GET /liveness/verify` - Liveness verification page
- `POST /liveness/verify` - Submit verification data
- `GET /liveness/results` - View verification results

### Dashboard
- `GET /` - User dashboard

## Security Features

- ✅ Password hashing with Werkzeug
- ✅ CSRF protection with Flask-WTF
- ✅ Secure session cookies (HTTPOnly, SameSite)
- ✅ SQL injection prevention with SQLAlchemy ORM
- ✅ Anti-spoofing detection with AI model
- ✅ Rate limiting ready (can be added per route)
- ✅ HTTPS support for production

## Troubleshooting

### Camera Access Not Working
- Ensure HTTPS is enabled (camera access requires secure context)
- Check browser permissions for camera access
- Try a different browser (Chrome/Firefox recommended)

### Model Loading Fails
- Verify `resnet_checkpoint_epoch_8.pt` exists in the `ai_models/` directory
- Check that PyTorch is properly installed
- Ensure correct path in `config.py`

### Database Errors
- Run `python run.py init_db` to initialize tables
- Check database file location and permissions
- Use `python run.py shell` to debug database state

### SSL Certificate Warnings
- This is expected for self-signed certificates in development
- Add an exception in your browser or use `https://localhost:5005` with trust bypass
- In production, use a valid SSL certificate from a certificate authority

## Performance Considerations

- Face detection calibration uses 30 frames for accuracy
- Liveness detection timeout is set to 10 seconds
- Frame processing is throttled to ~30 FPS
- AI model inference runs on CPU (GPU support can be added)

## Future Enhancements

- [ ] GPU acceleration for model inference
- [ ] Multi-language support
- [ ] API rate limiting
- [ ] Biometric template storage
- [ ] Advanced presentation attack detection
- [ ] Mobile app support
- [ ] Webhook notifications for verification events
- [ ] Admin dashboard for user management

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

For issues, questions, or suggestions, please [add contact/issue tracker information].

## Acknowledgments

- MediaPipe for face mesh detection
- PyTorch for deep learning framework
- Flask community for excellent documentation
