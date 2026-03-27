import os
from app import create_app, db, socketio
from app.models.user import User
from app.models.verification import Verification

app = create_app(os.getenv('FLASK_CONFIG') or 'default')

@app.shell_context_processor
def make_shell_context():
    """Make database models available in Flask shell"""
    return {'db': db, 'User': User, 'Verification': Verification}

@app.cli.command()
def init_db():
    """Initialize the database with tables"""
    db.create_all()
    print("Database initialized!")

@app.cli.command()
def create_admin():
    """Create an admin user"""
    admin = User(
        username='admin',
        email='admin@example.com',
        first_name='Admin',
        last_name='User',
        verification_level=2,
        is_verified=True
    )
    admin.set_password('admin123')
    
    db.session.add(admin)
    db.session.commit()
    print("Admin user created! (username: admin, password: admin123)")

if __name__ == '__main__':
    # Try to run with HTTPS first (required for webcam access)
    try:
        print("Starting server with HTTPS...")
        print("Access your app at: https://localhost:5005")
        print("Note: You'll need to accept the security warning for self-signed certificate")
        
        socketio.run(app, 
                    debug=True, 
                    host='0.0.0.0', 
                    port=5005,
                    ssl_context='adhoc')  # Auto-generate self-signed certificate
    except Exception as e:
        print(f"HTTPS failed: {e}")
        print("Trying to install pyOpenSSL...")
        
        # Try to install pyOpenSSL if not available
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyOpenSSL'])
            print("pyOpenSSL installed successfully. Please restart the application.")
        except Exception as install_error:
            print(f"Failed to install pyOpenSSL: {install_error}")
            print("Please install manually: pip install pyOpenSSL")
            print("Falling back to HTTP - webcam may not work in modern browsers")
            
            # Fallback to HTTP
            socketio.run(app, debug=True, host='0.0.0.0', port=5005)