"""
Chatbot module for PalmPay - Handles chatbot interactions and responses
"""
import json
import random
from datetime import datetime

class PalmPayChatbot:
    """Chatbot class for handling user inquiries about the palm payment system"""
    
    def __init__(self):
        """Initialize the chatbot with predefined responses"""
        self.greeting_phrases = [
            "Hello! How can I help you with PalmPay today?",
            "Hi there! I'm your PalmPay assistant. What can I do for you?",
            "Welcome to PalmPay support! How may I assist you?"
        ]
        
        self.farewell_phrases = [
            "Thank you for using PalmPay. Have a great day!",
            "Is there anything else I can help you with? If not, have a wonderful day!",
            "Thanks for chatting with PalmPay support. Come back if you have more questions!"
        ]
        
        # Knowledge base of responses
        self.responses = {
            "palm_enrollment": [
                "To enroll your palm print, go to your account settings and select 'Palm Enrollment'. Follow the on-screen instructions to complete the process.",
                "Palm enrollment is easy! Just navigate to the Palm Enrollment page from your dashboard and follow the guided steps.",
                "Make sure you're in a well-lit environment when enrolling your palm print for the best results."
            ],
            "payment_security": [
                "PalmPay uses advanced biometric security with palm print recognition and 128-bit encryption to protect your transactions.",
                "Your palm print data is securely stored and never leaves our secure servers. We use advanced encryption to protect all data.",
                "Our system verifies that only your palm can access your account, preventing unauthorized access."
            ],
            "transaction_issues": [
                "If you're experiencing issues with a transaction, please check your transaction history for details, then contact our support team.",
                "For delayed transactions, please allow up to 24 hours for processing. If it's still pending after that, contact support.",
                "You can dispute a transaction from your transaction history page by selecting the transaction and clicking 'Dispute'."
            ],
            "account_access": [
                "If you can't access your account, try resetting your password. If you still have issues, contact our support team.",
                "Account access requires either your password or palm authentication. Make sure your palm is properly enrolled.",
                "For security reasons, repeated failed authentication attempts may temporarily lock your account."
            ],
            "palm_recognition": [
                "Our palm recognition technology uses deep learning and pattern recognition to identify the unique features of your palm.",
                "Make sure to place your palm flat in front of the camera, and ensure there's good lighting for optimal recognition.",
                "Palm recognition is more secure than fingerprints because it captures more data points and is harder to spoof."
            ]
        }
        
        # Default responses when no specific topic is matched
        self.default_responses = [
            "I'm not sure I understand. Could you rephrase your question about PalmPay?",
            "I don't have information on that topic yet. Is there something else about PalmPay I can help with?",
            "Hmm, that's outside my knowledge area. I can help with enrollment, payments, security, and account issues."
        ]
    
    def get_greeting(self):
        """Return a random greeting"""
        return random.choice(self.greeting_phrases)
    
    def get_farewell(self):
        """Return a random farewell message"""
        return random.choice(self.farewell_phrases)
    
    def get_response(self, user_message):
        """
        Process user message and return appropriate response
        
        Args:
            user_message (str): The user's message text
            
        Returns:
            dict: Response with message and timestamp
        """
        user_message = user_message.lower()
        
        # Check for greetings
        if any(word in user_message for word in ["hello", "hi", "hey", "greetings"]):
            response_text = self.get_greeting()
        
        # Check for farewells
        elif any(word in user_message for word in ["bye", "goodbye", "see you", "thanks", "thank you"]):
            response_text = self.get_farewell()
        
        # Check for specific topics
        elif any(word in user_message for word in ["enroll", "register", "sign up", "palm print"]):
            response_text = random.choice(self.responses["palm_enrollment"])
        
        elif any(word in user_message for word in ["security", "safe", "protect", "secure"]):
            response_text = random.choice(self.responses["payment_security"])
        
        elif any(word in user_message for word in ["transaction", "payment", "purchase", "buy"]):
            response_text = random.choice(self.responses["transaction_issues"])
        
        elif any(word in user_message for word in ["access", "login", "account", "sign in"]):
            response_text = random.choice(self.responses["account_access"])
        
        elif any(word in user_message for word in ["palm", "recognition", "scan", "verify"]):
            response_text = random.choice(self.responses["palm_recognition"])
            
        # Default response if no specific topic is matched
        else:
            response_text = random.choice(self.default_responses)
        
        return {
            "message": response_text,
            "timestamp": datetime.now().strftime("%H:%M")
        }

# Create a global instance of the chatbot
chatbot = PalmPayChatbot()

def get_chatbot_response(message):
    """
    Get a response from the chatbot
    
    Args:
        message (str): User's message
        
    Returns:
        str: JSON string with the response message and timestamp
    """
    response = chatbot.get_response(message)
    return json.dumps(response)