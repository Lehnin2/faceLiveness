from datetime import datetime
from app import db

class Verification(db.Model):
    __tablename__ = 'verifications'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Verification details
    verification_type = db.Column(db.String(50), nullable=False)  # 'liveness', 'biometric', 'document'
    challenge_type = db.Column(db.String(50))  # 'blink', 'look_left', 'look_right'
    
    # Results
    success = db.Column(db.Boolean, nullable=False)
    ai_confidence = db.Column(db.Float)  # AI model confidence score
    failure_reason = db.Column(db.String(200))
    
    # Metadata
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    session_id = db.Column(db.String(100))
    
    # Processing details
    processing_time = db.Column(db.Float)  # Time taken in seconds
    frames_processed = db.Column(db.Integer)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    def mark_completed(self, success, ai_confidence=None, failure_reason=None):
        """Mark verification as completed"""
        self.success = success
        self.ai_confidence = ai_confidence
        self.failure_reason = failure_reason
        self.completed_at = datetime.utcnow()
        
        # Calculate processing time
        if self.created_at:
            self.processing_time = (self.completed_at - self.created_at).total_seconds()
        
        db.session.commit()
    
    def to_dict(self):
        """Convert verification to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'verification_type': self.verification_type,
            'challenge_type': self.challenge_type,
            'success': self.success,
            'ai_confidence': self.ai_confidence,
            'failure_reason': self.failure_reason,
            'processing_time': self.processing_time,
            'frames_processed': self.frames_processed,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    @staticmethod
    def get_user_success_rate(user_id, verification_type=None):
        """Get user's success rate for verifications"""
        query = Verification.query.filter_by(user_id=user_id)
        
        if verification_type:
            query = query.filter_by(verification_type=verification_type)
        
        total = query.count()
        if total == 0:
            return 0.0
        
        successful = query.filter_by(success=True).count()
        return (successful / total) * 100
    
    @staticmethod
    def get_recent_attempts(user_id, limit=5):
        """Get recent verification attempts"""
        return Verification.query.filter_by(user_id=user_id)\
            .order_by(Verification.created_at.desc())\
            .limit(limit).all()
    
    def __repr__(self):
        return f'<Verification {self.verification_type} - {self.success}>'