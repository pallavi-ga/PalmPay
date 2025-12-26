from app import app, db
from routes import *
from create_admin import create_admin_user

# Create admin user if it doesn't exist
with app.app_context():
    create_admin_user()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
