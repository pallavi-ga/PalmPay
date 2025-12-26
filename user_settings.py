"""
User settings module for the Palm Print Authentication System
Handles user profile management, security settings, and preferences
"""

import os
import secrets
import logging
from PIL import Image
from io import BytesIO
import base64
from datetime import datetime
from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
from models import User, PalmPrint, AuthenticationLog

logger = logging.getLogger(__name__)

class UserSettings:
    """Class for managing user settings and profile information"""
    
    @staticmethod
    def update_profile(user_id, username=None, email=None, first_name=None, 
                      last_name=None, phone=None):
        """
        Update user profile information
        
        Args:
            user_id (int): User ID
            username (str, optional): New username
            email (str, optional): New email
            first_name (str, optional): New first name
            last_name (str, optional): New last name
            phone (str, optional): New phone number
            
        Returns:
            tuple: (bool, str) - (success, message)
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return False, "User not found"
            
            # Check if username is taken (if being changed)
            if username and username != user.username:
                existing_user = User.query.filter_by(username=username).first()
                if existing_user:
                    return False, "Username is already taken"
                user.username = username
            
            # Check if email is taken (if being changed)
            if email and email != user.email:
                existing_user = User.query.filter_by(email=email).first()
                if existing_user:
                    return False, "Email is already registered"
                user.email = email
            
            # Update other fields if provided
            if first_name:
                user.first_name = first_name
            if last_name:
                user.last_name = last_name
            if phone:
                user.phone = phone
            
            db.session.commit()
            return True, "Profile updated successfully"
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating profile: {str(e)}")
            return False, f"Error updating profile: {str(e)}"
    
    @staticmethod
    def change_password(user_id, current_password, new_password):
        """
        Change user password
        
        Args:
            user_id (int): User ID
            current_password (str): Current password
            new_password (str): New password
            
        Returns:
            tuple: (bool, str) - (success, message)
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return False, "User not found"
            
            # Verify current password
            if not check_password_hash(user.password_hash, current_password):
                return False, "Current password is incorrect"
            
            # Validate new password
            if len(new_password) < 8:
                return False, "New password must be at least 8 characters long"
            
            # Update password
            user.password_hash = generate_password_hash(new_password)
            db.session.commit()
            
            logger.info(f"Password changed for user {user.id}")
            return True, "Password changed successfully"
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error changing password: {str(e)}")
            return False, f"Error changing password: {str(e)}"
    
    @staticmethod
    def save_profile_picture(user_id, image_data):
        """
        Save user profile picture
        
        Args:
            user_id (int): User ID
            image_data (str): Base64 encoded image data
            
        Returns:
            tuple: (bool, str) - (success, message or filename)
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return False, "User not found"
            
            # Parse base64 image
            image_format, image_data = image_data.split(';base64,')
            image_ext = image_format.split('/')[-1]
            
            # Generate unique filename
            filename = f"user_{user_id}_{secrets.token_hex(8)}.{image_ext}"
            filepath = os.path.join(current_app.static_folder, 'uploads', filename)
            
            # Ensure uploads directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Decode and process image
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            
            # Resize image to standard size
            img.thumbnail((300, 300))
            
            # Save image
            img.save(filepath)
            
            # Update user profile
            user.profile_picture = filename
            db.session.commit()
            
            return True, filename
            
        except Exception as e:
            logger.error(f"Error saving profile picture: {str(e)}")
            return False, f"Error saving profile picture: {str(e)}"
    
    @staticmethod
    def export_user_data(user_id, include_auth_logs=True, include_palm_data=False):
        """
        Export user data in structured format
        
        Args:
            user_id (int): User ID
            include_auth_logs (bool): Whether to include authentication logs
            include_palm_data (bool): Whether to include palm print data
            
        Returns:
            dict: User data in dictionary format
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return None
            
            # Base user data
            user_data = {
                'user_info': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'phone': user.phone,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'is_admin': user.is_admin,
                    'has_palm_print': user.has_palm_print
                }
            }
            
            # Add authentication logs if requested
            if include_auth_logs:
                user_data['authentication_logs'] = [{
                    'id': log.id,
                    'timestamp': log.timestamp.isoformat(),
                    'success': log.success,
                    'similarity_score': log.similarity_score
                } for log in user.auth_logs]
            
            # Add palm print data if requested (and if user is admin or self-export)
            if include_palm_data and user.has_palm_print:
                palm_prints = []
                for palm in user.palm_prints:
                    palm_data = {
                        'id': palm.id,
                        'created_at': palm.created_at.isoformat(),
                    }
                    # Only include the actual embedding if necessary (e.g., for research)
                    # palm_data['embedding'] = palm.get_embedding()
                    palm_prints.append(palm_data)
                
                user_data['palm_prints'] = palm_prints
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error exporting user data: {str(e)}")
            return None
    
    @staticmethod
    def get_auth_statistics(user_id):
        """
        Get authentication statistics for a user
        
        Args:
            user_id (int): User ID
            
        Returns:
            dict: Authentication statistics
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return None
            
            total_auths = len(user.auth_logs)
            successful_auths = sum(1 for log in user.auth_logs if log.success)
            success_rate = (successful_auths / total_auths * 100) if total_auths > 0 else 0
            
            # Get average similarity score for successful authentications
            successful_scores = [log.similarity_score for log in user.auth_logs 
                               if log.success and log.similarity_score is not None]
            avg_score = sum(successful_scores) / len(successful_scores) if successful_scores else 0
            
            # Get recent authentication trends (last 10 logs)
            recent_logs = user.get_recent_auth_logs(10)
            recent_success_rate = (sum(1 for log in recent_logs if log.success) / len(recent_logs) * 100) if recent_logs else 0
            
            return {
                'total_authentications': total_auths,
                'successful_authentications': successful_auths,
                'failed_authentications': total_auths - successful_auths,
                'success_rate': success_rate,
                'average_similarity_score': avg_score,
                'recent_success_rate': recent_success_rate,
                'last_authentication': user.auth_logs[-1].timestamp.isoformat() if user.auth_logs else None,
            }
            
        except Exception as e:
            logger.error(f"Error getting auth statistics: {str(e)}")
            return None