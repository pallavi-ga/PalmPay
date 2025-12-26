"""
Palm Payment Module for the Palm Payment System
Handles payment processing using palm print recognition
"""

import os
import uuid
import logging
import json
from datetime import datetime
import numpy as np
from app import db
from models import User, PalmPrint, Transaction, AuthenticationLog
from palm_recognition import preprocess_palm_image, extract_features, calculate_cosine_similarity, verify_palm_identity

logger = logging.getLogger(__name__)

class PalmPayment:
    """Class for handling palm-based payment processing"""
    
    @staticmethod
    def process_payment(user_id, amount, merchant, description, palm_image_data=None, payment_pin=None):
        """
        Process a payment using palm print verification and PIN
        
        Args:
            user_id (int): User ID
            amount (float): Payment amount
            merchant (str): Merchant name
            description (str): Payment description
            palm_image_data (str, optional): Base64 encoded palm image for verification
            payment_pin (str, optional): 6-digit PIN for payment verification
            
        Returns:
            tuple: (bool, str, dict) - (success, message, transaction data)
        """
        from models import AuthenticationLog
        try:
            # Verify user exists and has sufficient balance
            user = User.query.get(user_id)
            if not user:
                failed_log = AuthenticationLog(user_id=user_id, success=False, similarity_score=0.0)
                db.session.add(failed_log)
                db.session.commit()
                return False, "User not found", None
                
            if user.wallet_balance < amount:
                failed_log = AuthenticationLog(user_id=user_id, success=False, similarity_score=0.0)
                db.session.add(failed_log)
                db.session.commit()
                return False, "Insufficient funds", None
            
            # Verify palm print if provided
            if palm_image_data:
                palm_print = PalmPrint.query.filter_by(user_id=user_id).first()
                if not palm_print:
                    failed_log = AuthenticationLog(user_id=user_id, success=False, similarity_score=0.0)
                    db.session.add(failed_log)
                    db.session.commit()
                    return False, "Palm print not enrolled", None
                
                # Decode and process palm image
                image_data = palm_image_data.split(',')[1]
                
                # Convert to numpy array
                import base64
                nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                
                # Preprocess and extract features - now returns tensor, original image, and processed image
                img_tensor, original_img, processed_img = preprocess_palm_image(nparr)
                palm_embedding = extract_features(img_tensor)
                
                # Save the palm images for this payment transaction (they're stored with the user ID)
                palm_print.save_palm_images(original_img, processed_img)
                
                # Get all palm prints for enhanced security verification
                all_palm_prints = PalmPrint.query.all()
                
                # Enhanced verification that checks against all palm prints
                # This prevents a user from accessing another user's account with their palm
                is_authenticated, similarity_score, matched_user_id, error_message = verify_palm_identity(
                    palm_embedding, 
                    user_id, 
                    all_palm_prints,
                    threshold=0.85
                )
                
                # Log potential fraud attempts (palm belongs to a different user)
                if matched_user_id and matched_user_id != user_id and similarity_score >= 0.7:
                    failed_log = AuthenticationLog(user_id=user_id, success=False, similarity_score=similarity_score)
                    db.session.add(failed_log)
                    db.session.commit()
                    return False, "Payment verification failed due to security check. Please contact support.", None
                
                # Payment fails if palm verification fails
                if not is_authenticated:
                    failed_log = AuthenticationLog(user_id=user_id, success=False, similarity_score=similarity_score)
                    db.session.add(failed_log)
                    db.session.commit()
                    return False, f"Palm print verification failed. Similarity score: {similarity_score:.2f}", None
                
                logger.info(f"Palm verification successful for payment. Score: {similarity_score:.2f}")
            
            # Verify payment PIN if provided
            if payment_pin:
                # Check if PIN is 6 digits
                if not payment_pin.isdigit() or len(payment_pin) != 6:
                    failed_log = AuthenticationLog(user_id=user_id, success=False, similarity_score=similarity_score)
                    db.session.add(failed_log)
                    db.session.commit()
                    return False, "Invalid PIN format. Please enter a 6-digit PIN.", None
                
                # Check if PIN matches user's PIN
                if not user.payment_pin or user.payment_pin != payment_pin:
                    failed_log = AuthenticationLog(user_id=user_id, success=False, similarity_score=similarity_score)
                    db.session.add(failed_log)
                    db.session.commit()
                    return False, "Incorrect payment PIN. Please try again.", None
                
                logger.info(f"Payment PIN verification successful for user {user_id}")
            else:
                # Do not require PIN if not provided (first step)
                pass
            
            # Generate a unique reference ID for the transaction
            reference_id = str(uuid.uuid4())
            
            # Create the transaction
            transaction = Transaction(
                user_id=user_id,
                amount=amount,
                merchant=merchant,
                description=description,
                transaction_type='payment',
                status='completed',
                reference_id=reference_id
            )
            
            # Update user's wallet balance
            user.wallet_balance -= amount
            
            # Save changes to database
            db.session.add(transaction)
            db.session.commit()
            
            # If we reach here, authentication and payment succeeded: update the log to success
            success_log = AuthenticationLog(user_id=user_id, success=True, similarity_score=similarity_score if palm_image_data else 1.0)
            db.session.add(success_log)
            db.session.commit()
            
            # Return transaction details
            transaction_data = {
                'id': transaction.id,
                'reference_id': transaction.reference_id,
                'amount': transaction.amount,
                'merchant': transaction.merchant,
                'description': transaction.description,
                'timestamp': transaction.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'new_balance': user.wallet_balance
            }
            
            return True, "Payment processed successfully", transaction_data
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing payment: {str(e)}")
            failed_log = AuthenticationLog(user_id=user_id, success=False, similarity_score=0.0)
            db.session.add(failed_log)
            db.session.commit()
            return False, f"Payment processing error: {str(e)}", None
    
    @staticmethod
    def get_transaction_history(user_id, limit=10):
        """
        Get transaction history for a user
        
        Args:
            user_id (int): User ID
            limit (int, optional): Maximum number of transactions to retrieve
            
        Returns:
            list: List of transaction dictionaries
        """
        try:
            transactions = Transaction.query.filter_by(user_id=user_id).order_by(
                Transaction.timestamp.desc()).limit(limit).all()
            
            transaction_list = []
            for t in transactions:
                transaction_list.append({
                    'id': t.id,
                    'reference_id': t.reference_id,
                    'amount': t.amount,
                    'merchant': t.merchant,
                    'description': t.description,
                    'type': t.transaction_type,
                    'status': t.status,
                    'timestamp': t.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            return transaction_list
            
        except Exception as e:
            logger.error(f"Error getting transaction history: {str(e)}")
            return []
    
    @staticmethod
    def get_transaction_stats(user_id):
        """
        Get transaction statistics for a user
        
        Args:
            user_id (int): User ID
            
        Returns:
            dict: Transaction statistics
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return None
                
            total_transactions = Transaction.query.filter_by(user_id=user_id).count()
            
            # Calculate total spent
            total_spent = db.session.query(db.func.sum(Transaction.amount)).filter(
                Transaction.user_id == user_id,
                Transaction.transaction_type == 'payment'
            ).scalar() or 0
            
            # Get top merchants
            from sqlalchemy import func
            merchant_counts = db.session.query(
                Transaction.merchant, func.count(Transaction.id).label('count')
            ).filter(
                Transaction.user_id == user_id
            ).group_by(
                Transaction.merchant
            ).order_by(
                func.count(Transaction.id).desc()
            ).limit(3).all()
            
            top_merchants = [{'name': m.merchant, 'count': m.count} for m in merchant_counts]
            
            return {
                'total_transactions': total_transactions,
                'total_spent': float(total_spent),
                'current_balance': user.wallet_balance,
                'top_merchants': top_merchants,
                'has_payment_method': bool(user.card_last_four)
            }
            
        except Exception as e:
            logger.error(f"Error getting transaction stats: {str(e)}")
            return None