"""
Database Migration Script for Palm Print Authentication System
This script updates the database schema to include new columns in the User model
"""
import os
import sys
import logging
import psycopg2
from psycopg2 import sql

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection using environment variables."""
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST"),
        database=os.environ.get("PGDATABASE"),
        user=os.environ.get("PGUSER"),
        password=os.environ.get("PGPASSWORD"),
        port=os.environ.get("PGPORT")
    )
    conn.autocommit = True
    return conn

def check_column_exists(conn, table_name, column_name):
    """Check if a column exists in a table."""
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s AND column_name = %s;
        """, (table_name, column_name))
        return cursor.fetchone() is not None

def add_column(conn, table_name, column_name, column_type):
    """Add a column to a table if it doesn't exist."""
    with conn.cursor() as cursor:
        if not check_column_exists(conn, table_name, column_name):
            query = sql.SQL("ALTER TABLE {} ADD COLUMN {} {}").format(
                sql.Identifier(table_name),
                sql.Identifier(column_name),
                sql.SQL(column_type)
            )
            cursor.execute(query)
            logger.info(f"Added column '{column_name}' to table '{table_name}'")
            return True
        else:
            logger.info(f"Column '{column_name}' already exists in table '{table_name}'")
            return False

def migrate_database():
    """Perform the database migration."""
    logger.info("Starting database migration...")
    
    try:
        conn = get_db_connection()
        
        # Add new columns to the user table for profile
        add_column(conn, "user", "last_login", "TIMESTAMP")
        add_column(conn, "user", "first_name", "VARCHAR(64)")
        add_column(conn, "user", "last_name", "VARCHAR(64)")
        add_column(conn, "user", "phone", "VARCHAR(20)")
        add_column(conn, "user", "profile_picture", "VARCHAR(256)")
        
        # Add payment related columns to user table
        add_column(conn, "user", "wallet_balance", "FLOAT DEFAULT 1000.00 NOT NULL")
        add_column(conn, "user", "payment_pin", "VARCHAR(6)")
        add_column(conn, "user", "card_last_four", "VARCHAR(4)")
        add_column(conn, "user", "card_brand", "VARCHAR(20)")
        add_column(conn, "user", "card_expiry", "VARCHAR(7)")
        
        # Add image path columns to palm_print table
        add_column(conn, "palm_print", "original_image_path", "VARCHAR(256)")
        add_column(conn, "palm_print", "processed_image_path", "VARCHAR(256)")
        
        # Create transactions table if it doesn't exist
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transaction (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES "user" (id),
                    amount FLOAT NOT NULL,
                    description VARCHAR(255) NOT NULL,
                    merchant VARCHAR(100) NOT NULL,
                    transaction_type VARCHAR(20) NOT NULL DEFAULT 'payment',
                    status VARCHAR(20) NOT NULL DEFAULT 'completed',
                    reference_id VARCHAR(64) UNIQUE NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
        logger.info("Created transaction table successfully")
        
        conn.close()
        logger.info("Database migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during database migration: {str(e)}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1)