import sqlite3
import os

def add_generated_documents_column():
    try:
        # Connect to the database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check current table structure
        cursor.execute("PRAGMA table_info(users)")
        columns_before = cursor.fetchall()
        print("Current table structure:")
        for col in columns_before:
            print(f"  - {col[1]} ({col[2]})")
        
        # Check if column already exists
        column_names = [col[1] for col in columns_before]
        if 'generated_documents' in column_names:
            print("❌ Column 'generated_documents' already exists!")
            conn.close()
            return
        
        # Add the new column
        cursor.execute('ALTER TABLE users ADD COLUMN generated_documents TEXT')
        conn.commit()
        print("✅ Successfully added 'generated_documents' column")
        
        # Verify the column was added
        cursor.execute("PRAGMA table_info(users)")
        columns_after = cursor.fetchall()
        print("\nUpdated table structure:")
        for col in columns_after:
            print(f"  - {col[1]} ({col[2]})")
        
        # Confirm the new column exists
        new_column_names = [col[1] for col in columns_after]
        if 'generated_documents' in new_column_names:
            print("✅ Column addition verified successfully!")
        else:
            print("❌ Column verification failed!")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    add_generated_documents_column()