# app/liveness/routes.py - Liveness Detection Routes
import time
from flask import render_template, request, jsonify, session
from flask_login import login_required, current_user
from flask_socketio import emit, join_room, leave_room
import cv2
import base64
import numpy as np
from app.liveness import bp
from app import socketio, db
from app.models.verification import Verification
from .detector import LivenessDetector
import uuid

# Store active sessions
active_sessions = {}

@bp.route('/verify')
@login_required
def verify():
    """Render liveness verification page"""
    return render_template('liveness/verify.html')

@socketio.on('start_verification')
def handle_start_verification():
    """Handle start of verification session"""
    session_id = str(uuid.uuid4())
    session['verification_id'] = session_id
    
    # Create new detector instance
    detector = LivenessDetector()
    active_sessions[session_id] = detector
    
    join_room(session_id)
    emit('verification_started', {
        'session_id': session_id,
        'challenge': detector.get_current_challenge()
    })

@socketio.on('process_frame')
def handle_process_frame(data):
    """Handle incoming video frame for processing"""
    session_id = session.get('verification_id')
    if not session_id or session_id not in active_sessions:
        emit('error', {'message': 'Invalid session'})
        return
        
    detector = active_sessions[session_id]
    
    # Remove this line - you don't need session_id from client
    # session_id = data.get('session_id')  # Remove this
    
    # ADD SERVER-SIDE THROTTLING
    current_time = time.time()
    if hasattr(detector, 'last_server_process_time'):
        if current_time - detector.last_server_process_time < 0.033:  # 30 FPS limit
            return
    detector.last_server_process_time = current_time
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        status, message, is_complete, processed_frame = detector.process_frame(frame)
        
        # Encode processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get AI status
        ai_status, ai_color = detector.get_ai_status()
        
        emit('frame_processed', {
            'status': status,
            'message': message,
            'is_complete': is_complete,
            'processed_frame': f'data:image/jpeg;base64,{processed_b64}',
            'ai_status': ai_status,
            'challenge': detector.get_current_challenge()
        })
        
        # Handle completion
        if is_complete:
            # Log verification result
            verification = Verification(
                user_id=current_user.id,
                verification_type='liveness',
                challenge_type=detector.get_current_challenge(),
                success=status == 'verified',
                ai_confidence=0.9 if status == 'verified' else 0.1
            )
            db.session.add(verification)
            db.session.commit()
            
            # Clean up session
            if session_id in active_sessions:
                del active_sessions[session_id]
                
            emit('verification_complete', {
                'result': status,
                'message': message
            })
            
    except Exception as e:
        emit('error', {'message': f'Processing error: {str(e)}'})


@socketio.on('stop_verification')
def handle_stop_verification():
    """Handle stopping verification session"""
    session_id = session.get('verification_id')
    if session_id and session_id in active_sessions:
        del active_sessions[session_id]
        leave_room(session_id)
        emit('verification_stopped')

@bp.route('/api/verification_history')
@login_required
def verification_history():
    """Get user's verification history"""
    verifications = Verification.query.filter_by(user_id=current_user.id)\
                                    .order_by(Verification.created_at.desc())\
                                    .limit(10).all()
    
    history = []
    for v in verifications:
        history.append({
            'id': v.id,
            'challenge_type': v.challenge_type,
            'result': 'verified' if v.success else 'failed',  # Convert boolean to string
            'timestamp': v.created_at.isoformat(),  # Use created_at but call it timestamp
            'ai_confidence': v.ai_confidence
        })
    
    return jsonify(history)