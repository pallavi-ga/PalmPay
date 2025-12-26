"""
Script to create admin user for PalmPay
"""

import os
import sys
from app import app, db
from models import User
from werkzeug.security import generate_password_hash

def create_admin_user():
    """Create an admin user if it doesn't exist"""
    with app.app_context():
        # Check if admin user exists
        admin = User.query.filter_by(username='admin').first()
        if admin:
            print("Admin user already exists.")
            return
        
        # Create admin user
        admin_user = User(
            username='admin',
            email='admin@example.com',
            password_hash=generate_password_hash('admin123', method='pbkdf2:sha256'),
            is_admin=True,
            first_name='Admin',
            last_name='User',
            wallet_balance=1000.00
        )
        
        db.session.add(admin_user)
        db.session.commit()
        print("Admin user created successfully!")

if __name__ == "__main__":
    create_admin_user()