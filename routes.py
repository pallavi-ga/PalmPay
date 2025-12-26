import io
import os
import base64
import logging
from datetime import datetime
import numpy as np
from flask import render_template, redirect, url_for, request, flash, jsonify, session
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from app import app, db
from models import User, PalmPrint, AuthenticationLog, Transaction
from palm_recognition import preprocess_palm_image, extract_features, calculate_cosine_similarity, verify_palm_identity
from chatbot import get_chatbot_response
import uuid
import traceback
import random
from flask_mail import Mail, Message
import string
import pytz

logger = logging.getLogger(__name__)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'hsbhumika20@gmail.com'
app.config['MAIL_PASSWORD'] = 'xaxb iwvi oxgg ghaq'
mail = Mail(app)

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def utc_to_india(utc_dt):
    india = pytz.timezone('Asia/Kolkata')
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt)
    return utc_dt.astimezone(india)

# Sample product list (in a real app, use a database)
PRODUCTS = [
    {"id": 1, "name": "Premium Laptop", "price": 1299.99, "desc": "15\" Display, 16GB RAM", "icon": "laptop"},
    {"id": 2, "name": "Wireless Headphones", "price": 249.99, "desc": "Noise-canceling, Bluetooth 5.0", "icon": "headphones"},
    {"id": 3, "name": "Wireless Mouse", "price": 49.99, "desc": "Ergonomic Design, Long Battery Life", "icon": "mouse"}
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        payment_pin = request.form.get('payment_pin')
        first_name = request.form.get('first_name', '')
        last_name = request.form.get('last_name', '')
        
        # Check if username or email already exists
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists. Please choose another one.', 'danger')
            return redirect(url_for('register'))
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already registered. Please use another email.', 'danger')
            return redirect(url_for('register'))
        
        # Set admin flag if this is the admin account
        is_admin = False
        if email == 'admin@example.com' and username == 'admin':
            is_admin = True
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            password_hash=generate_password_hash(password),
            payment_pin=payment_pin,  # Store the payment PIN
            is_admin=is_admin,
            wallet_balance=1000.00  # Initial balance for testing
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please enroll your palm print.', 'success')
        login_user(new_user)
        return redirect(url_for('palm_enrollment'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        # Redirect to appropriate dashboard based on user type
        if current_user.is_admin:
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            # For admin user, redirect directly to admin dashboard
            if user.is_admin:
                flash('Welcome, Administrator!', 'success')
                return redirect(url_for('admin_dashboard'))
            
            # For regular users, check palm print enrollment
            palm_print = PalmPrint.query.filter_by(user_id=user.id).first()
            
            if palm_print:
                # Redirect to dashboard if user has palm print enrolled
                return redirect(url_for('dashboard'))
            else:
                # Redirect to palm enrollment if user doesn't have palm print enrolled
                flash('Please enroll your palm print for biometric authentication.', 'info')
                return redirect(url_for('palm_enrollment'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get authentication logs for current user
    auth_logs = AuthenticationLog.query.filter_by(user_id=current_user.id).order_by(AuthenticationLog.timestamp.desc()).limit(10).all()
    
    # Calculate success rate
    total_auths = len(auth_logs)
    successful_auths = sum(1 for log in auth_logs if log.success)
    success_rate = (successful_auths / total_auths * 100) if total_auths > 0 else 0
    
    return render_template('dashboard.html', 
                           auth_logs=auth_logs, 
                           total_auths=total_auths,
                           successful_auths=successful_auths,
                           success_rate=success_rate)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all users
    users = User.query.all()
    
    # Get recent authentication logs
    auth_logs = AuthenticationLog.query.order_by(AuthenticationLog.timestamp.desc()).limit(20).all()
    
    # Convert timestamps to India time for display
    for log in auth_logs:
        log.india_time = utc_to_india(log.timestamp)
    
    # Calculate overall system stats
    total_users = User.query.count()
    enrolled_users = db.session.query(User.id).join(PalmPrint).distinct().count()
    total_auths = AuthenticationLog.query.count()
    successful_auths = AuthenticationLog.query.filter_by(success=True).count()
    success_rate = (successful_auths / total_auths * 100) if total_auths > 0 else 0
    
    # Get transaction data
    from models import Transaction
    recent_transactions = Transaction.query.order_by(Transaction.timestamp.desc()).limit(10).all()
    total_transactions = Transaction.query.count()
    total_transaction_amount = db.session.query(db.func.sum(Transaction.amount)).scalar() or 0
    avg_transaction = total_transaction_amount / total_transactions if total_transactions > 0 else 0
    transaction_success_rate = 100 * (Transaction.query.filter_by(status='completed').count() / total_transactions) if total_transactions > 0 else 0
    
    # Security monitoring - Detect suspicious activity
    # Filter for failed authentications with high similarity scores (potential fraud attempts)
    suspicious_auths = AuthenticationLog.query.filter(
        AuthenticationLog.success == False # Count all failed attempts as suspicious for dashboard alert
    ).order_by(AuthenticationLog.timestamp.desc()).limit(10).all()
    
    # Get multiple failed attempts from same user within short time periods
    from sqlalchemy import func, and_, text
    repeated_failures = db.session.query(
        AuthenticationLog.user_id,
        func.count(AuthenticationLog.id).label('failure_count')
    ).filter(
        AuthenticationLog.success == False,
        AuthenticationLog.timestamp > text("NOW() - INTERVAL '24 HOURS'")
    ).group_by(
        AuthenticationLog.user_id
    ).having(
        func.count(AuthenticationLog.id) >= 3  # 3+ failures in 24 hours
    ).all()
    
    # Get users with suspicious stats
    users_with_alerts = []
    for user in users:
        # Calculate user's authentication success rate
        user_auths = AuthenticationLog.query.filter_by(user_id=user.id).count()
        if user_auths >= 5:  # Only check users with sufficient history
            user_success_rate = (AuthenticationLog.query.filter_by(
                user_id=user.id, success=True).count() / user_auths) * 100
            
            # Flag users with unusually low success rates
            if user_success_rate < 60:
                alert = {
                    'user_id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'success_rate': user_success_rate,
                    'auth_count': user_auths,
                    'alert_type': 'Low Authentication Success Rate',
                    'severity': 'warning'
                }
                users_with_alerts.append(alert)
    
    # Get transaction stats for chart
    transaction_stats = db.session.query(
        db.func.date_trunc('day', Transaction.timestamp).label('day'),
        db.func.sum(Transaction.amount).label('sum')
    ).group_by('day').order_by('day').limit(7).all()
    
    transaction_labels = [stat.day.strftime("%m/%d") for stat in transaction_stats]
    transaction_volumes = [float(stat.sum) for stat in transaction_stats]
    
    # Get potential security incidents
    suspicious_auth_attempts = AuthenticationLog.query.filter(
        AuthenticationLog.success == False # Count all failed attempts for the alert banner
    ).count()
    
    return render_template('admin_dashboard.html',
                           users=users,
                           auth_logs=auth_logs,
                           total_users=total_users,
                           enrolled_users=enrolled_users,
                           total_auths=total_auths,
                           successful_auths=successful_auths,
                           success_rate=success_rate,
                           recent_transactions=recent_transactions,
                           total_transactions=total_transactions,
                           total_transaction_amount=total_transaction_amount,
                           avg_transaction=avg_transaction,
                           transaction_success_rate=transaction_success_rate,
                           transaction_labels=transaction_labels,
                           transaction_volumes=transaction_volumes,
                           suspicious_auth_attempts=suspicious_auth_attempts,
                           suspicious_auths=suspicious_auths,
                           repeated_failures=repeated_failures,
                           users_with_alerts=users_with_alerts)

@app.route('/palm_enrollment', methods=['GET', 'POST'])
@login_required
def palm_enrollment():
    # Check if user already has a palm print enrolled
    existing_palm = PalmPrint.query.filter_by(user_id=current_user.id).first()
    
    if existing_palm:
        flash('You already have a palm print enrolled. You can re-enroll if needed.', 'info')
    
    if request.method == 'POST':
        # Get the palm image data from the request
        palm_image_data = request.json.get('palm_image')
        
        if not palm_image_data:
            return jsonify({'success': False, 'message': 'No palm image data received'})
        
        try:
            # Decode the base64 image
            image_data = base64.b64decode(palm_image_data.split(',')[1])
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Preprocess the palm image - now returns tensor, original image, and processed image
            img_tensor, original_img, processed_img = preprocess_palm_image(nparr)
            
            # Extract features using the pre-trained model
            palm_embedding = extract_features(img_tensor)
            
            # Create new palm print record or update existing one
            if existing_palm:
                existing_palm.set_embedding(palm_embedding)
                # Save the palm images
                existing_palm.save_palm_images(original_img, processed_img)
                db.session.commit()
                logger.info(f"Updated palm print for user {current_user.id}")
            else:
                new_palm_print = PalmPrint(user_id=current_user.id)
                new_palm_print.set_embedding(palm_embedding)
                # Save the palm images
                new_palm_print.save_palm_images(original_img, processed_img)
                db.session.add(new_palm_print)
                db.session.commit()
                logger.info(f"Enrolled new palm print for user {current_user.id}")
            
            return jsonify({'success': True, 'message': 'Palm print enrolled successfully'})
            
        except Exception as e:
            logger.error(f"Error in palm enrollment: {str(e)}")
            return jsonify({'success': False, 'message': f'Error processing palm image: {str(e)}'})
    
    return render_template('palm_enrollment.html')

@app.route('/palm_authentication', methods=['GET', 'POST'])
@login_required
def palm_authentication():
    # Ensure user has enrolled their palm print
    enrolled_palm = PalmPrint.query.filter_by(user_id=current_user.id).first()
    
    if not enrolled_palm:
        flash('You need to enroll your palm print first.', 'warning')
        return redirect(url_for('palm_enrollment'))
    
    if request.method == 'POST':
        print("Received POST request for palm authentication")
        palm_image_data = request.json.get('palm_image')
        
        if not palm_image_data:
            print("No palm image data received")
            return jsonify({'success': False, 'message': 'No palm image data received'})
        
        try:
            print("Attempting to process image")
            # Decode the base64 image
            image_data = base64.b64decode(palm_image_data.split(',')[1])
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Preprocess the palm image - now returns tensor, original image, and processed image
            img_tensor, original_img, processed_img = preprocess_palm_image(nparr)
            
            print("Image processed, attempting feature extraction")
            # Extract features using the pre-trained model
            new_palm_embedding = extract_features(img_tensor)
            
            print("Features extracted, attempting verification")
            # Get the stored palm embedding
            stored_palm_embedding = np.array(enrolled_palm.get_embedding())
            
            # Get all palm prints for enhanced security verification
            all_palm_prints = PalmPrint.query.all()
            
            # Verify palm identity securely (checks against all stored palm prints)
            # Only authenticates if the palm matches ONLY the current user's enrolled palm
            is_authenticated, similarity_score, matched_user_id = verify_palm_identity(
                new_palm_embedding, 
                current_user.id, 
                all_palm_prints,
                threshold=0.85
            )
            
            print(f"Verification result: authenticated={is_authenticated}, score={similarity_score}")
            
            # Security alert if there's a potential palm print fraud attempt
            if matched_user_id and matched_user_id != current_user.id and similarity_score >= 0.7:
                logger.warning(f"Potential palm print fraud! User {current_user.id} attempted authentication but matched user {matched_user_id} with score {similarity_score}")
            
            # Log the authentication attempt
            auth_log = AuthenticationLog(
                user_id=current_user.id,
                success=is_authenticated,
                similarity_score=float(similarity_score)
            )
            db.session.add(auth_log)
            db.session.commit()
            
            print("Authentication log saved to database")
            
            logger.info(f"Authentication attempt for user {current_user.id}: score={similarity_score}, success={is_authenticated}")
            
            # Return the authentication result
            if is_authenticated:
                # Redirect to appropriate dashboard based on user type
                redirect_url = url_for('admin_dashboard') if current_user.is_admin else url_for('dashboard')
                return jsonify({
                    'success': True, 
                    'authenticated': True, 
                    'message': 'Authentication successful',
                    'similarity_score': float(similarity_score),
                    'redirect_url': redirect_url
                })
            else:
                return jsonify({
                    'success': True, 
                    'authenticated': False, 
                    'message': 'Authentication failed. Please try again or use password login.',
                    'similarity_score': float(similarity_score)
                })
                
        except Exception as e:
            print(f"An error occurred: {e}")
            logger.error(f"Error in palm authentication: {str(e)}")
            return jsonify({'success': False, 'message': f'Error processing palm image: {str(e)}'})
    
    return render_template('palm_authentication.html')

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Delete palm prints
    PalmPrint.query.filter_by(user_id=user.id).delete()
    
    # Delete authentication logs
    AuthenticationLog.query.filter_by(user_id=user.id).delete()
    
    # Delete user
    db.session.delete(user)
    db.session.commit()
    
    flash(f'User {user.username} has been deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/user/<int:user_id>/reset_palm', methods=['POST'])
@login_required
def reset_palm(user_id):
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    
    # Delete palm prints
    PalmPrint.query.filter_by(user_id=user.id).delete()
    db.session.commit()
    
    flash(f'Palm print data for user {user.username} has been reset.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/palm_prints')
@login_required
def admin_palm_prints():
    """Admin panel to view all palm prints with processed images"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all palm prints with user data
    palm_prints = db.session.query(PalmPrint, User).join(User).all()
    
    return render_template('admin_palm_prints.html', palm_prints=palm_prints)

@app.route('/admin/user/<int:user_id>/palm_details')
@login_required
def admin_user_palm_details(user_id):
    """Detailed view of a user's palm prints"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    palm_prints = PalmPrint.query.filter_by(user_id=user_id).all()
    
    # Get the authentication logs for this user
    auth_logs = AuthenticationLog.query.filter_by(user_id=user_id).order_by(
        AuthenticationLog.timestamp.desc()).limit(10).all()
    
    return render_template('admin_user_palm_details.html', 
                          user=user, 
                          palm_prints=palm_prints,
                          auth_logs=auth_logs)

@app.route('/about')
def about():
    """About page with information about the biometric system"""
    return render_template('about.html')

@app.route('/features')
def features():
    """Features page showcasing system capabilities"""
    return render_template('features.html')

@app.route('/contact')
def contact():
    """Contact page with form to reach administrators"""
    return render_template('contact.html')

@app.route('/privacy')
@app.route('/privacy_policy')
def privacy_policy():
    """Privacy policy page"""
    return render_template('privacy.html')

@app.route('/terms')
def terms_of_service():
    """Terms of service page"""
    return render_template('terms.html')

@app.route('/faq')
def faq():
    """Frequently asked questions page"""
    return render_template('faq.html')

@app.route('/security')
def security():
    """Security information page about palm authentication"""
    return render_template('security.html')

@app.route('/user_profile')
@login_required
def user_profile():
    """User profile page with settings and authentication history"""
    # Get authentication logs for current user
    auth_logs = current_user.get_recent_auth_logs(10)
    
    # Calculate success rate and stats
    total_auths = len(current_user.auth_logs)
    successful_auths = sum(1 for log in current_user.auth_logs if log.success)
    success_rate = (successful_auths / total_auths * 100) if total_auths > 0 else 0
    
    # Get transaction history
    from palm_payment import PalmPayment
    transactions = PalmPayment.get_transaction_history(current_user.id)
    transaction_stats = PalmPayment.get_transaction_stats(current_user.id)
    
    # Convert timestamps to India time for display
    for log in auth_logs:
        log.india_time = utc_to_india(log.timestamp)
    
    return render_template('user_profile.html',
                          auth_logs=auth_logs,
                          total_auths=total_auths,
                          successful_auths=successful_auths,
                          success_rate=success_rate,
                          transactions=transactions,
                          transaction_stats=transaction_stats)

@app.route('/shop')
def shop():
    return render_template('shop.html', products=PRODUCTS)

@app.route('/cart')
@login_required
def cart():
    cart = session.get('cart', {})
    cart_items = []
    subtotal = 0
    for pid, qty in cart.items():
        product = next((p for p in PRODUCTS if p['id'] == int(pid)), None)
        if product:
            item = product.copy()
            item['quantity'] = qty
            item['total'] = qty * product['price']
            cart_items.append(item)
            subtotal += item['total']
    shipping = 9.99 if subtotal > 0 else 0
    tax = round(subtotal * 0.06, 2)
    order_total = subtotal + shipping + tax
    return render_template('cart.html', cart_items=cart_items, subtotal=subtotal, shipping=shipping, tax=tax, order_total=order_total)

@app.route('/cart/add/<int:product_id>')
@login_required
def add_to_cart(product_id):
    cart = session.get('cart', {})
    cart[str(product_id)] = cart.get(str(product_id), 0) + 1
    session['cart'] = cart
    return redirect(url_for('cart'))

@app.route('/cart/remove/<int:product_id>')
@login_required
def remove_from_cart(product_id):
    cart = session.get('cart', {})
    if str(product_id) in cart:
        del cart[str(product_id)]
        session['cart'] = cart
    return redirect(url_for('cart'))

@app.route('/cart/update/<int:product_id>/<action>')
@login_required
def update_cart_quantity(product_id, action):
    cart = session.get('cart', {})
    pid = str(product_id)
    if pid in cart:
        if action == 'inc':
            cart[pid] += 1
        elif action == 'dec':
            cart[pid] = max(1, cart[pid] - 1)
        session['cart'] = cart
    return redirect(url_for('cart'))

@app.route('/payments/checkout')
@login_required
def checkout():
    cart = session.get('cart', {})
    cart_items = []
    subtotal = 0
    cart_count = 0
    for pid, qty in cart.items():
        product = next((p for p in PRODUCTS if p['id'] == int(pid)), None)
        if product:
            item = product.copy()
            item['quantity'] = qty
            item['total'] = qty * product['price']
            cart_items.append(item)
            subtotal += item['total']
            cart_count += qty
    shipping = 9.99 if subtotal > 0 else 0
    tax = round(subtotal * 0.06, 2)
    order_total = subtotal + shipping + tax
    session['order_total'] = order_total
    session['cart_count'] = cart_count
    session['subtotal'] = subtotal
    session['shipping'] = shipping
    session['tax'] = tax
    return render_template('checkout.html', cart_items=cart_items, subtotal=subtotal, shipping=shipping, tax=tax, order_total=order_total)

@app.route('/payments/method')
@login_required
def payment_method():
    """Payment method selection page"""
    cart = session.get('cart', {})
    cart_items = []
    subtotal = 0
    for pid, qty in cart.items():
        product = next((p for p in PRODUCTS if p['id'] == int(pid)), None)
        if product:
            item = product.copy()
            item['quantity'] = qty
            item['total'] = qty * product['price']
            cart_items.append(item)
            subtotal += item['total']
    shipping = 9.99 if subtotal > 0 else 0
    tax = round(subtotal * 0.06, 2)
    order_total = subtotal + shipping + tax
    return render_template('payment_method.html', cart_items=cart_items, subtotal=subtotal, shipping=shipping, tax=tax, order_total=order_total)

@app.route('/payments/palm')
@login_required
def palm_payment_page():
    """Palm payment processing page"""
    # Get amount from session or query parameter
    amount = request.args.get('amount', 0.0, type=float)
    return render_template('palm_payment.html', amount=amount)

@app.route('/payments/process', methods=['POST'])
@login_required
def process_payment():
    """Process a payment with palm verification and PIN"""
    if not request.is_json:
        return jsonify({'success': False, 'message': 'Invalid request format'})
    try:
        # Get payment details from request
        data = request.json
        amount = float(data.get('amount', 0.0))
        merchant = data.get('merchant', 'Online Store')
        description = data.get('description', 'Online Purchase')
        palm_image_data = data.get('palm_image')
        payment_pin = data.get('payment_pin')
        # Save order details before clearing session
        session['last_order'] = {
            'cart_count': session.get('cart_count', 0),
            'subtotal': session.get('subtotal', 0),
            'shipping': session.get('shipping', 0),
            'tax': session.get('tax', 0),
            'order_total': session.get('order_total', 0),
            'amount_paid': amount,
        }
        # Step 1: Palm verification only
        if palm_image_data and not payment_pin:
            from palm_payment import PalmPayment
            user = current_user
            palm_print = PalmPrint.query.filter_by(user_id=user.id).first()
            if not palm_print:
                from models import AuthenticationLog
                failed_log = AuthenticationLog(user_id=user.id, success=False, similarity_score=0.0)
                db.session.add(failed_log)
                db.session.commit()
                return jsonify({'success': False, 'message': 'Palm print not enrolled'})
            try:
                image_data = palm_image_data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                img_tensor, original_img, processed_img = preprocess_palm_image(nparr)
                palm_embedding = extract_features(img_tensor)
                all_palm_prints = PalmPrint.query.all()
                is_authenticated, similarity_score, matched_user_id, error_message = verify_palm_identity(
                    palm_embedding, 
                    user.id, 
                    all_palm_prints,
                    threshold=0.90
                )
                if not is_authenticated:
                    from models import AuthenticationLog
                    failed_log = AuthenticationLog(user_id=user.id, success=False, similarity_score=similarity_score)
                    db.session.add(failed_log)
                    db.session.commit()
                    return jsonify({'success': False, 'message': error_message, 'similarity_score': similarity_score})
                return jsonify({'success': True, 'palm_verified': True, 'similarity_score': similarity_score})
            except Exception as e:
                from models import AuthenticationLog
                failed_log = AuthenticationLog(user_id=user.id, success=False, similarity_score=0.0)
                db.session.add(failed_log)
                db.session.commit()
                return jsonify({'success': False, 'message': f'Palm image processing failed: {str(e)}'})
        # Step 2: Palm and PIN verification
        if palm_image_data and payment_pin:
            from palm_payment import PalmPayment
            success, message, transaction_data = PalmPayment.process_payment(
                current_user.id, amount, merchant, description, palm_image_data, payment_pin
            )
            if success:
                return jsonify({
                    'success': True, 
                    'message': message, 
                    'transaction': transaction_data,
                    'redirect_url': url_for('payment_success')
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': message
                })
        return jsonify({'success': False, 'message': 'Palm image and PIN are required.'})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Error processing payment: {str(e)}'})

@app.route('/payments/success')
@login_required
def payment_success():
    """Payment success page"""
    order = session.pop('last_order', None)
    if not order:
        # Try to get the latest OTP payment transaction for the user
        from models import Transaction
        tx = Transaction.query.filter_by(user_id=current_user.id, merchant='PalmPay OTP').order_by(Transaction.timestamp.desc()).first()
        if tx:
            order = {
                'amount_paid': tx.amount,
                'cart_count': 1,
                'subtotal': tx.amount,
                'shipping': 0,
                'tax': 0,
                'order_total': tx.amount,
                'transaction_id': tx.reference_id,
                'payment_method': 'OTP via Email',
                'date': tx.timestamp
            }
    return render_template('payment_success.html', now=datetime.now(), order=order)

@app.route('/payments/failed')
@login_required
def payment_failed():
    """Payment failed page"""
    error_message = request.args.get('error', 'An error occurred during payment processing.')
    return render_template('payment_failed.html', error_message=error_message)

@app.route('/wallet')
@login_required
def wallet():
    """User wallet and transaction history page"""
    from palm_payment import PalmPayment
    transactions = PalmPayment.get_transaction_history(current_user.id, 20)
    transaction_stats = PalmPayment.get_transaction_stats(current_user.id)
    
    return render_template('wallet.html', 
                          transactions=transactions,
                          transaction_stats=transaction_stats)

@app.route('/wallet/download-history')
@login_required
def download_transaction_history():
    """Download transaction history as CSV file"""
    import csv
    from io import StringIO
    from flask import make_response
    from palm_payment import PalmPayment
    
    # Get all transactions for the current user
    transactions = PalmPayment.get_transaction_history(current_user.id, limit=100)
    
    # Create a CSV file in memory
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header row
    writer.writerow(['Transaction ID', 'Reference', 'Date', 'Amount', 'Merchant', 'Description', 'Type', 'Status'])
    
    # Write transaction data
    for transaction in transactions:
        writer.writerow([
            transaction['id'],
            transaction['reference_id'],
            transaction['timestamp'],
            f"${transaction['amount']:.2f}",
            transaction['merchant'],
            transaction['description'],
            transaction['type'],
            transaction['status']
        ])
    
    # Create response with CSV file
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=palmpay_transactions_{datetime.now().strftime('%Y%m%d')}.csv"
    response.headers["Content-type"] = "text/csv"
    
    return response

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """API endpoint for the chatbot"""
    if not request.is_json:
        return jsonify({'error': 'Invalid request format. JSON required.'}), 400
    
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400
    
    # Get response from chatbot
    response = get_chatbot_response(user_message)
    
    return response, 200, {'Content-Type': 'application/json'}

@app.route('/add_funds', methods=['POST'])
@login_required
def add_funds():
    try:
        amount = float(request.form.get('amount', 0))
        if amount <= 0:
            flash('Please enter a valid amount to add.', 'danger')
            return redirect(url_for('wallet'))
        current_user.wallet_balance += amount
        # Record the deposit as a transaction
        reference_id = str(uuid.uuid4())
        deposit_tx = Transaction(
            user_id=current_user.id,
            amount=amount,
            merchant='PalmPay Wallet',
            description='Funds Added',
            transaction_type='deposit',
            status='completed',
            reference_id=reference_id
        )
        db.session.add(deposit_tx)
        db.session.commit()
        flash(f'Successfully added ${amount:.2f} to your wallet!', 'success')
    except Exception as e:
        flash(f'Error adding funds: {str(e)}', 'danger')
    return redirect(url_for('wallet'))

@app.route('/payments/start-palm', methods=['POST'])
@login_required
def start_palm_payment():
    try:
        amount = float(request.form.get('order_total', 0))
        session['order_total'] = amount
    except Exception:
        session['order_total'] = 0
    return redirect(url_for('palm_payment_page'))

@app.route('/test_fail_log')
@login_required
def test_fail_log():
    from models import AuthenticationLog
    from app import db
    import random
    failed_log = AuthenticationLog(
        user_id=current_user.id,
        success=False,
        similarity_score=random.uniform(0.0, 0.5)
    )
    db.session.add(failed_log)
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/payments/otp', methods=['GET'])
@login_required
def otp_payment_page():
    amount = session.get('order_total', 0.0)
    return render_template('otp_payment.html', amount=amount, email=current_user.email)

@app.route('/payments/send-otp', methods=['POST'])
@login_required
def send_otp():
    otp = generate_otp()
    session['otp'] = otp
    session['otp_attempts'] = 0
    msg = Message('Your Payment OTP', sender=app.config['MAIL_USERNAME'], recipients=[current_user.email])
    msg.body = f'Your OTP for payment is: {otp}. It is valid for 10 minutes.'
    mail.send(msg)
    return 'OTP sent', 200

@app.route('/payments/verify-otp', methods=['POST'])
@login_required
def verify_otp():
    from palm_payment import PalmPayment
    data = request.form
    entered_otp = data.get('otp')
    amount = session.get('order_total', 0.0)
    if not entered_otp or 'otp' not in session:
        flash('OTP expired or not sent. Please try again.', 'danger')
        return redirect(url_for('otp_payment_page'))
    if session.get('otp_attempts', 0) >= 5:
        flash('Too many incorrect attempts. Please request a new OTP.', 'danger')
        session.pop('otp', None)
        return redirect(url_for('otp_payment_page'))
    if entered_otp == session['otp']:
        # Use PalmPayment.process_payment for wallet deduction and transaction
        success, message, transaction_data = PalmPayment.process_payment(
            current_user.id, float(amount), 'PalmPay OTP', 'Payment via OTP', None, None
        )
        session.pop('otp', None)
        if success:
            flash('Payment successful via OTP!', 'success')
            return redirect(url_for('payment_success'))
        else:
            flash(message, 'danger')
            return redirect(url_for('otp_payment_page'))
    else:
        session['otp_attempts'] = session.get('otp_attempts', 0) + 1
        flash('Invalid OTP. Please try again.', 'danger')
        return redirect(url_for('otp_payment_page'))
