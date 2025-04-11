import sqlite3
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def investigate_product_categories():
    """
    Investigate product categories in the database
    """
    # Database path
    db_path = 'data/retail_inventory.db'
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get unique categories
    cursor.execute("SELECT DISTINCT Category FROM products")
    categories = cursor.fetchall()
    
    print("Unique Categories:")
    for category in categories:
        print(f"- {category[0]}")
    
    # Count products per category
    cursor.execute("""
        SELECT Category, COUNT(*) as product_count 
        FROM products 
        GROUP BY Category
    """)
    category_counts = cursor.fetchall()
    
    print("\nProducts per Category:")
    for category, count in category_counts:
        print(f"- {category}: {count} products")
    
    # Check for data in demand_forecasting
    cursor.execute("""
        SELECT DISTINCT p.Category, COUNT(df.Product_ID) as record_count
        FROM products p
        LEFT JOIN demand_forecasting df ON p.Product_ID = df.Product_ID
        GROUP BY p.Category
    """)
    category_data_counts = cursor.fetchall()
    
    print("\nDemand Forecasting Records per Category:")
    for category, count in category_data_counts:
        print(f"- {category}: {count} records")
    
    conn.close()

# Call the function
investigate_product_categories()