import sqlite3
import os
import sys
import traceback

def initialize_database():
    try:
        # Ensure the data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Connect to the database (creates it if it doesn't exist)
        conn = sqlite3.connect('data/retail_inventory.db')
        cursor = conn.cursor()
        
        # Read the schema file
        try:
            with open('database_schema.sql', 'r') as f:
                schema = f.read()
        except FileNotFoundError:
            print("Error: database_schema.sql file not found!")
            return False
        
        # Execute the schema
        try:
            conn.executescript(schema)
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error executing schema: {e}")
            traceback.print_exc()
            return False
        
        # Verify table creation
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Tables created:")
        for table in tables:
            print(f"- {table[0]}")
        
        conn.close()
        print("Database initialized successfully.")
        return True
    
    except Exception as e:
        print(f"Unexpected error during database initialization: {e}")
        traceback.print_exc()
        return False

def populate_initial_data():
    """
    Optional method to populate initial data if needed
    """
    try:
        conn = sqlite3.connect('data/retail_inventory.db')
        cursor = conn.cursor()
        
        # Example of inserting initial data
        # You can expand this with more comprehensive data seeding
        
        # Sample data for inventory_monitoring
        cursor.executemany('''
        INSERT OR REPLACE INTO inventory_monitoring 
        (Product_ID, Store_ID, Stock_Levels, Supplier_Lead_Time_days, 
        Stockout_Frequency, Reorder_Point, Warehouse_Capacity) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', [
            (1, 1, 100, 7, 0, 50, 1000),
            (2, 1, 200, 5, 1, 75, 1500),
            # Add more initial inventory records as needed
        ])
        
        conn.commit()
        conn.close()
        print("Initial data populated successfully.")
        return True
    
    except Exception as e:
        print(f"Error populating initial data: {e}")
        traceback.print_exc()
        return False

def main():
    # Initialize database
    if initialize_database():
        # Optionally populate with initial data
        populate_initial_data()
    else:
        print("Database initialization failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()