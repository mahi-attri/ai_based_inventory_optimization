import sqlite3
import os
import random
import numpy as np
from datetime import datetime, timedelta
import json
import logging

def initialize_database(db_path: str):
    """
    Initialize the SQLite database
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        SQLite database connection
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create or connect to the database
        conn = sqlite3.connect(db_path)
        
        # Create a comprehensive database with synthetic data
        create_comprehensive_database(conn)
        
        logger.info(f"Database initialized successfully at {db_path}")
        
        return conn
    
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def alter_table_add_column(cursor, table_name, column_name, column_type):
    """
    Safely add a column to a table if it doesn't exist
    
    Args:
        cursor: SQLite cursor
        table_name: Name of the table
        column_name: Name of the column to add
        column_type: SQL type of the column
    """
    try:
        # Check if column exists
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add column if it doesn't exist
        if column_name not in columns:
            alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            cursor.execute(alter_query)
            print(f"Added column {column_name} to {table_name}")
    except sqlite3.OperationalError as e:
        print(f"Error checking/adding column {column_name}: {e}")

def create_comprehensive_database(conn):
    """
    Create a comprehensive database with synthetic data
    
    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()
    
    # Database schema creation
    schema_creation_queries = [
        '''CREATE TABLE IF NOT EXISTS products (
            Product_ID INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Category TEXT NOT NULL,
            Subcategory TEXT,
            Unit_Cost REAL NOT NULL,
            MSRP REAL NOT NULL,
            Weight_kg REAL,
            Volume_m3 REAL,
            Perishable INTEGER NOT NULL DEFAULT 0,
            Supplier_ID INTEGER,
            Lead_Time_Days INTEGER DEFAULT 7,
            Active INTEGER NOT NULL DEFAULT 1
        )''',
        
        '''CREATE TABLE IF NOT EXISTS inventory_monitoring (
            Product_ID INTEGER,
            Store_ID INTEGER,
            Stock_Levels INTEGER NOT NULL DEFAULT 0,
            Supplier_Lead_Time_days INTEGER NOT NULL DEFAULT 7,
            Stockout_Frequency INTEGER NOT NULL DEFAULT 0,
            Reorder_Point INTEGER NOT NULL DEFAULT 50,
            Expiry_Date TEXT,
            Warehouse_Capacity INTEGER NOT NULL DEFAULT 1000,
            Order_Fulfillment_Time_days INTEGER NOT NULL DEFAULT 3,
            PRIMARY KEY (Product_ID, Store_ID)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS demand_forecasting (
            Product_ID INTEGER,
            Store_ID INTEGER,
            Date TEXT,
            Sales_Quantity INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (Product_ID, Store_ID, Date)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS pricing_optimization (
            Product_ID INTEGER,
            Store_ID INTEGER,
            Price REAL NOT NULL DEFAULT 0.0,
            Competitor_Prices TEXT,
            Discounts REAL NOT NULL DEFAULT 0.0,
            Storage_Cost REAL NOT NULL DEFAULT 0.1,
            Elasticity_Index REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (Product_ID, Store_ID)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS stores (
            Store_ID INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Location TEXT NOT NULL,
            Size_sqm INTEGER,
            Type TEXT,
            Open_Date TEXT,
            Active INTEGER NOT NULL DEFAULT 1
        )''',
        
        '''CREATE TABLE IF NOT EXISTS suppliers (
            Supplier_ID INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Contact_Email TEXT,
            Contact_Phone TEXT,
            Address TEXT,
            Reliability_Score REAL DEFAULT 0.8,
            Active INTEGER NOT NULL DEFAULT 1
        )'''
    ]
    
    # Execute schema creation
    for query in schema_creation_queries:
        cursor.execute(query)
    
    # Add missing columns to demand_forecasting
    columns_to_add = [
        ("Price", "REAL NOT NULL DEFAULT 0.0"),
        ("Promotions", "REAL NOT NULL DEFAULT 0.0"),
        ("Seasonality_Factors", "REAL NOT NULL DEFAULT 1.0"),
        ("External_Factors", "REAL NOT NULL DEFAULT 1.0"),
        ("Demand_Trend", "REAL NOT NULL DEFAULT 0.0"),
        ("Customer_Segments", "TEXT")
    ]
    
    for column_name, column_type in columns_to_add:
        alter_table_add_column(cursor, "demand_forecasting", column_name, column_type)
    
    # Add Sales_Volume column to pricing_optimization
    alter_table_add_column(cursor, "pricing_optimization", "Sales_Volume", "INTEGER DEFAULT 0")
    alter_table_add_column(cursor, "pricing_optimization", "Customer_Reviews", "REAL DEFAULT 3.5")
    alter_table_add_column(cursor, "pricing_optimization", "Return_Rate", "REAL DEFAULT 0.05")
    
    # Commit table structure changes
    conn.commit()
    
    # Generate synthetic data
    np.random.seed(42)
    
    # You would then proceed with generating synthetic data as in the previous implementation
    # This would include populating stores, suppliers, products, demand forecasting, 
    # pricing optimization, and inventory monitoring tables
    
    # Insert Stores
    store_data = _generate_store_data(10)
    cursor.executemany('''
    INSERT OR REPLACE INTO stores
    (Store_ID, Name, Location, Size_sqm, Type, Open_Date, Active)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', store_data)
    
    # Insert Suppliers
    supplier_data = _generate_supplier_data(20)
    cursor.executemany('''
    INSERT OR REPLACE INTO suppliers
    (Supplier_ID, Name, Contact_Email, Contact_Phone, Address, Reliability_Score, Active)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', supplier_data)
    
    # Insert Products
    product_data = _generate_product_data(100)
    cursor.executemany('''
    INSERT OR REPLACE INTO products 
    (Product_ID, Name, Category, Subcategory, Unit_Cost, MSRP, 
    Weight_kg, Volume_m3, Perishable, Supplier_ID, Lead_Time_Days, Active)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', product_data)
    
    # Insert Demand Forecasting Data
    demand_data = _generate_demand_data(100, 10)
    cursor.executemany('''
    INSERT OR REPLACE INTO demand_forecasting 
    (Product_ID, Store_ID, Date, Sales_Quantity, 
    Price, Promotions, Seasonality_Factors, 
    External_Factors, Demand_Trend, Customer_Segments)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', demand_data)
    
    # Insert Pricing Optimization Data
    pricing_data = _generate_pricing_data(100, 10, cursor)
    cursor.executemany('''
    INSERT OR REPLACE INTO pricing_optimization 
    (Product_ID, Store_ID, Price, Competitor_Prices, 
    Discounts, Storage_Cost, Elasticity_Index, Sales_Volume,
    Customer_Reviews, Return_Rate)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', pricing_data)
    
    # Insert Inventory Monitoring Data
    inventory_data = _generate_inventory_data(100, 10, cursor)
    cursor.executemany('''
    INSERT OR REPLACE INTO inventory_monitoring 
    (Product_ID, Store_ID, Stock_Levels, Supplier_Lead_Time_days, 
    Stockout_Frequency, Reorder_Point, Expiry_Date, 
    Warehouse_Capacity, Order_Fulfillment_Time_days)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', inventory_data)
    
    # Commit and close
    conn.commit()

def _generate_store_data(num_stores):
    """Generate synthetic store data"""
    store_data = []
    for store_id in range(1, num_stores + 1):
        store_types = ["Urban", "Suburban", "Rural", "Mall", "Express"]
        store_type = random.choice(store_types)
        
        # Generate store open date (between 1-5 years ago)
        days_ago = random.randint(365, 365 * 5)
        open_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        store_data.append((
            store_id,
            f"Store {store_id} - {store_type}",
            f"{random.randint(100, 999)} Example St, City {store_id}",
            random.randint(500, 5000),  # Size in square meters
            store_type,
            open_date,
            1  # Active
        ))
    return store_data

def _generate_supplier_data(num_suppliers):
    """Generate synthetic supplier data"""
    supplier_data = []
    for supplier_id in range(100, 100 + num_suppliers):
        supplier_data.append((
            supplier_id,
            f"Supplier {supplier_id}",
            f"supplier{supplier_id}@example.com",
            f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            f"{random.randint(100, 999)} Business St, Supplier City",
            round(random.uniform(0.6, 0.98), 2),  # Reliability score
            1  # Active
        ))
    return supplier_data

def _generate_product_data(num_products):
    """Generate synthetic product data"""
    product_data = []
    for product_id in range(1, num_products + 1):
        # Determine product category
        if product_id <= 25:  # Electronics
            category = "electronics"
            subcategory = random.choice(["Phones", "Laptops", "Tablets", "Accessories"])
            unit_cost = np.random.uniform(30, 400)
            msrp = unit_cost * np.random.uniform(1.2, 1.5)
        elif product_id <= 50:  # Clothing
            category = "clothing"
            subcategory = random.choice(["Shirts", "Pants", "Dresses", "Accessories"])
            unit_cost = np.random.uniform(10, 150)
            msrp = unit_cost * np.random.uniform(1.5, 2.0)
        elif product_id <= 75:  # Groceries
            category = "groceries"
            subcategory = random.choice(["Produce", "Dairy", "Bakery", "Canned"])
            unit_cost = np.random.uniform(1, 30)
            msrp = unit_cost * np.random.uniform(1.3, 1.7)
            perishable = 1
        else:  # Other products
            category = "other"
            subcategory = random.choice(["Home", "Garden", "Office", "Misc"])
            unit_cost = np.random.uniform(3, 80)
            msrp = unit_cost * np.random.uniform(1.4, 1.8)
        
        # Create product data
        name = f"{category.title()} Product {product_id}"
        weight = np.random.uniform(0.1, 10.0)
        volume = np.random.uniform(0.001, 0.5)
        perishable = 1 if category == "groceries" else 0
        supplier_id = 100 + (product_id % 20)  # 20 different suppliers
        lead_time = np.random.randint(3, 21)
        
        product_data.append((
            product_id, name, category, subcategory, 
            unit_cost, msrp, weight, volume,
            perishable, supplier_id, lead_time, 1
        ))
    return product_data

def _generate_demand_data(num_products, num_stores):
    """Generate synthetic demand forecasting data"""
    demand_data = []
    today = datetime.now()
    
    for product_id in range(1, num_products + 1):
        for store_id in range(1, num_stores + 1):
            for days_ago in range(90):  # 90 days of historical data
                sales_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                
                # Realistic sales with seasonal and external factors
                base_sales = max(0, int(np.random.normal(30, 15)))
                
                # Add weekly seasonality
                day_of_week = (today - timedelta(days=days_ago)).weekday()
                if day_of_week in [5, 6]:  # Weekend
                    seasonality = np.random.uniform(1.1, 1.5)
                else:
                    seasonality = np.random.uniform(0.8, 1.1)
                
                external_factor = np.random.uniform(0.8, 1.2)
                
                # Random promotions occasionally
                if random.random() < 0.1:  # 10% chance of promotion
                    promotion_percentage = np.random.uniform(0.1, 0.3)  # 10-30% promotion
                else:
                    promotion_percentage = 0
                
                # Price variation
                price = np.random.uniform(50, 200)
                
                sales_quantity = max(0, int(base_sales * seasonality * external_factor * (1 - promotion_percentage * 2)))
                
                # Customer segments as JSON string
                segments = json.dumps({
                    "regular": round(random.uniform(0.4, 0.6), 2),
                    "premium": round(random.uniform(0.2, 0.4), 2),
                    "budget": round(random.uniform(0.1, 0.3), 2)
                })
                
                demand_data.append((
                    product_id, store_id, sales_date, sales_quantity,
                    price, promotion_percentage, 
                    seasonality, external_factor, 
                    0.001 * (90 - days_ago),  # Small upward trend
                    segments
                ))
    
    return demand_data
def _generate_pricing_data(num_products, num_stores, cursor):
    """
    Generate synthetic pricing optimization data
    
    Args:
        num_products: Number of products to generate data for
        num_stores: Number of stores to generate data for
        cursor: Database cursor for additional queries
    
    Returns:
        List of pricing optimization data
    """
    pricing_data = []
    
    for product_id in range(1, num_products + 1):
        for store_id in range(1, num_stores + 1):
            # Get product details
            cursor.execute("SELECT Category, Unit_Cost, MSRP FROM products WHERE Product_ID = ?", (product_id,))
            result = cursor.fetchone()
            
            if not result:
                continue
                
            category, unit_cost, msrp = result
            
            # Calculate current price (slightly varied from MSRP)
            price = msrp * np.random.uniform(0.9, 1.1)
            
            # Generate competitor prices as JSON
            competitor_prices = json.dumps({
                f"competitor_{i}": msrp * np.random.uniform(0.85, 1.15)
                for i in range(1, 4)
            })
            
            # Determine sales volume based on historical data
            cursor.execute("""
            SELECT AVG(Sales_Quantity) FROM demand_forecasting 
            WHERE Product_ID = ? AND Store_ID = ? AND Date >= date('now', '-30 days')
            """, (product_id, store_id))
            result = cursor.fetchone()
            avg_sales = result[0] if result[0] is not None else random.randint(10, 100)
            
            # Other pricing metrics
            discount = np.random.uniform(0, 0.2)
            storage_cost = unit_cost * np.random.uniform(0.01, 0.05)  # Storage cost as % of unit cost
            elasticity = np.random.uniform(0.5, 2.5)  # Price elasticity of demand
            customer_reviews = round(np.random.uniform(2.5, 5.0), 1)  # 2.5-5.0 star rating
            return_rate = round(np.random.uniform(0.01, 0.15), 3)  # 1-15% return rate
            
            pricing_data.append((
                product_id, store_id, 
                price,  # Price
                competitor_prices,  # Competitor Prices
                discount,  # Discounts
                storage_cost,  # Storage Cost
                elasticity,  # Elasticity Index
                int(avg_sales),  # Sales Volume
                customer_reviews,  # Customer Reviews
                return_rate  # Return Rate
            ))
    
    return pricing_data

def _generate_inventory_data(num_products, num_stores, cursor):
    """
    Generate synthetic inventory monitoring data
    
    Args:
        num_products: Number of products to generate data for
        num_stores: Number of stores to generate data for
        cursor: Database cursor for additional queries
    
    Returns:
        List of inventory monitoring data
    """
    inventory_data = []
    
    for product_id in range(1, num_products + 1):
        for store_id in range(1, num_stores + 1):
            # Get product details and sales volume
            cursor.execute("""
            SELECT p.Category, p.Perishable, po.Sales_Volume 
            FROM products p
            JOIN pricing_optimization po ON p.Product_ID = po.Product_ID
            WHERE p.Product_ID = ? AND po.Store_ID = ?
            """, (product_id, store_id))
            result = cursor.fetchone()
            
            if not result:
                continue
                
            category, perishable, sales_volume = result
            sales_volume = sales_volume if sales_volume else 30
            
            # Calculate realistic inventory levels
            stock_levels = max(10, int(sales_volume * np.random.uniform(1.5, 4.0)))  # 1.5-4 weeks of supply
            
            # Get supplier lead time from products
            cursor.execute("SELECT Lead_Time_Days FROM products WHERE Product_ID = ?", (product_id,))
            lead_time = cursor.fetchone()[0]
            
            # Stockout frequency (random but higher for high-volume or perishable items)
            if sales_volume > 50 or perishable == 1:
                stockout_freq = np.random.randint(0, 5)
            else:
                stockout_freq = np.random.randint(0, 2)
            
            # Calculate reorder point based on lead time and sales volume
            safety_stock = sales_volume * 0.5  # Half a week of safety stock
            reorder_point = int((lead_time / 7) * sales_volume + safety_stock)
            
            # Warehouse capacity based on category
            if category == "electronics":
                warehouse_capacity = stock_levels * np.random.uniform(2.0, 3.0)
            elif category == "clothing":
                warehouse_capacity = stock_levels * np.random.uniform(3.0, 5.0)
            elif category == "groceries":
                warehouse_capacity = stock_levels * np.random.uniform(1.5, 2.5)
            else:
                warehouse_capacity = stock_levels * np.random.uniform(2.0, 4.0)
            
            # Expiry date only for perishable products
            expiry_date = None
            if perishable == 1:
                expiry_date = (datetime.now() + timedelta(days=np.random.randint(7, 60))).strftime("%Y-%m-%d")
            
            # Order fulfillment time (days) - slightly correlated with lead time but shorter
            fulfillment_time = max(1, int(lead_time * np.random.uniform(0.2, 0.5)))
            
            inventory_data.append((
                product_id, store_id, stock_levels, lead_time, 
                stockout_freq, reorder_point, expiry_date, 
                int(warehouse_capacity), fulfillment_time
            ))
    
    return inventory_data

def main():
    """
    Main function to set up the database
    """
    # Path to your database
    db_path = "data/retail_inventory.db"
    
    # Initialize database
    initialize_database(db_path)
    print(f"Database initialized with comprehensive synthetic data at {db_path}")

if __name__ == "__main__":
    main()