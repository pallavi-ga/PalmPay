# PalmPay - Palm Print Biometric Payment System

A cutting-edge biometric authentication and payment platform that provides advanced security through innovative palm print recognition and deep learning technologies.

## Overview

PalmPay is a secure payment system that uses palm print biometrics as the primary authentication method. The system leverages deep learning technology to extract unique features from users' palm prints and securely verify their identity before processing payments.

## Key Features

- **Advanced Palm Biometrics**: Utilizes deep learning for accurate palm print recognition
- **Enhanced Security**: Identifies both main palm lines (red) and secondary lines (green) for improved accuracy
- **Multi-factor Authentication**: Combines palm biometrics with traditional security measures
- **Secure Payments**: Process transactions after biometric verification
- **Transaction History**: View and download transaction history
- **Admin Dashboard**: Comprehensive monitoring of users, authentications, and security alerts
- **Palm Management**: Store and display both original and processed palm images
- **Fraud Detection**: Advanced algorithms to detect potential fraud attempts
- **Interactive Chatbot**: Floating assistance feature for user queries

## Technology Stack

- **Backend**: Python 3.11, Flask, SQLAlchemy
- **Database**: PostgreSQL
- **Machine Learning**: PyTorch, OpenCV, ResNet18 for feature extraction
- **Frontend**: HTML5, CSS3, Bootstrap, JavaScript
- **Authentication**: Flask-Login
- **Security**: Cross-user protection, abnormal activity detection

## Installation and Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file
4. Initialize the database:
   ```
   python migrate_db.py
   ```
5. Create an admin user:
   ```
   python create_admin.py
   ```
6. Run the application:
   ```
   python main.py
   ```

## Environment Variables

Create a `.env` file with the following variables:
```
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost/palmpay
PGHOST=localhost
PGUSER=username
PGPASSWORD=password
PGDATABASE=palmpay
PGPORT=5432

# Flask Configuration
FLASK_APP=main.py
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

## User Types

1. **Admin User**:
   - Username: admin
   - Password: admin123
   - Email: admin@example.com
   - Access to admin dashboard, user management, and security monitoring

2. **Regular User**:
   - Self-registration available
   - Access to palm enrollment, authentication, and payment processing

## Biometric Process Flow

1. **Enrollment**: Users capture their palm image for feature extraction
2. **Processing**: System extracts palm lines and creates feature embedding
3. **Storage**: Both original and processed images are stored securely
4. **Authentication**: User palm prints are compared against stored templates
5. **Verification**: System confirms identity using similarity threshold (85%)

## Security Features

- Cross-user protection to prevent accessing someone else's account
- Detection of fake palm prints through feature matching
- Security alerts for suspicious authentication attempts
- Tracking of repeated authentication failures
- Admin monitoring of security incidents

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ResNet18 pre-trained model from PyTorch
- OpenCV for image processing
- Flask framework and extensions


To Run a Project
Python main.py

admin user name: admin
password: 123456789




normal user name: virat
password: 123456789