from app import db, login_manager
from flask_login import UserMixin
from datetime import datetime
import json

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Add nullable=True to all new fields to avoid errors with existing records
    last_login = db.Column(db.DateTime, nullable=True)
    first_name = db.Column(db.String(64), nullable=True)
    last_name = db.Column(db.String(64), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    profile_picture = db.Column(db.String(256), nullable=True)
    # Payment information
    wallet_balance = db.Column(db.Float, default=1000.00, nullable=False)  # Give initial balance for testing
    payment_pin = db.Column(db.String(6), nullable=True)  # PIN for payment verification
    card_last_four = db.Column(db.String(4), nullable=True)
    card_brand = db.Column(db.String(20), nullable=True)
    card_expiry = db.Column(db.String(7), nullable=True)  # MM/YYYY
    # Relationships
    palm_prints = db.relationship('PalmPrint', backref='user', lazy=True)
    auth_logs = db.relationship('AuthenticationLog', backref='user', lazy=True)
    transactions = db.relationship('Transaction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    @property
    def has_palm_print(self):
        """Check if user has enrolled their palm print"""
        return len(self.palm_prints) > 0
    
    @property
    def full_name(self):
        """Return user's full name if available, otherwise username"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
    
    @property
    def auth_success_rate(self):
        """Calculate authentication success rate"""
        total = len(self.auth_logs)
        if total == 0:
            return 0
        successful = sum(1 for log in self.auth_logs if log.success)
        return (successful / total) * 100
    
    def get_recent_auth_logs(self, limit=10):
        """Get the most recent authentication logs"""
        return AuthenticationLog.query.filter_by(user_id=self.id).order_by(
            AuthenticationLog.timestamp.desc()).limit(limit).all()

    def __repr__(self):
        return f'<User {self.username}>'

class PalmPrint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    palm_embedding = db.Column(db.Text, nullable=False)  # JSON string of embedding vector
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    original_image_path = db.Column(db.String(256), nullable=True)  # Path to original palm image
    processed_image_path = db.Column(db.String(256), nullable=True)  # Path to processed palm image
    
    def set_embedding(self, embedding_array):
        """Convert numpy array to JSON string for storage"""
        self.palm_embedding = json.dumps(embedding_array.tolist())
    
    def get_embedding(self):
        """Convert stored JSON string back to list"""
        return json.loads(self.palm_embedding)
    
    def save_palm_images(self, original_image, processed_image=None):
        """
        Save original and processed palm images to filesystem
        
        Args:
            original_image: Original palm image data (numpy array)
            processed_image: Processed palm image that shows detected palm lines (numpy array)
        """
        import cv2
        import os
        
        # Create unique filename based on user ID and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = f"palm_{self.user_id}_{timestamp}_original.png"
        processed_filename = f"palm_{self.user_id}_{timestamp}_processed.png"
        
        # Save original image
        original_path = os.path.join('static', 'palm_images', 'original', original_filename)
        cv2.imwrite(original_path, original_image)
        self.original_image_path = original_path
        
        # Save processed image if provided
        if processed_image is not None:
            processed_path = os.path.join('static', 'palm_images', 'processed', processed_filename)
            cv2.imwrite(processed_path, processed_image)
            self.processed_image_path = processed_path
    
    def __repr__(self):
        return f'<PalmPrint for User #{self.user_id}>'

class AuthenticationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    success = db.Column(db.Boolean, nullable=False)
    similarity_score = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        status = "Success" if self.success else "Failed"
        return f'<Auth {status} for User #{self.user_id} at {self.timestamp}>'
        
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(255), nullable=False)
    merchant = db.Column(db.String(100), nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False, default='payment')  # payment, refund, etc.
    status = db.Column(db.String(20), nullable=False, default='completed')  # completed, pending, failed
    reference_id = db.Column(db.String(64), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Transaction {self.reference_id}: ${self.amount:.2f} for User #{self.user_id} at {self.timestamp}>'
