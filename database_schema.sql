# database_schema.sql

-- Main tables from CSV data
CREATE TABLE IF NOT EXISTS demand_forecasting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Product_ID INTEGER NOT NULL,
    Date TEXT NOT NULL,
    Store_ID INTEGER NOT NULL,
    Sales_Quantity INTEGER NOT NULL,
    Price REAL NOT NULL,
    Promotions TEXT,
    Seasonality_Factors TEXT,
    External_Factors TEXT,
    Demand_Trend TEXT,
    Customer_Segments TEXT
);

CREATE TABLE IF NOT EXISTS inventory_monitoring (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Product_ID INTEGER NOT NULL,
    Store_ID INTEGER NOT NULL,
    Stock_Levels INTEGER NOT NULL,
    Supplier_Lead_Time_days INTEGER NOT NULL,
    Stockout_Frequency INTEGER NOT NULL,
    Reorder_Point INTEGER NOT NULL,
    Expiry_Date TEXT,
    Warehouse_Capacity INTEGER NOT NULL,
    Order_Fulfillment_Time_days INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing_optimization (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Product_ID INTEGER NOT NULL,
    Store_ID INTEGER NOT NULL,
    Price REAL NOT NULL,
    Competitor_Prices REAL,
    Discounts REAL,
    Sales_Volume INTEGER NOT NULL,
    Customer_Reviews INTEGER,
    Return_Rate REAL,
    Storage_Cost REAL NOT NULL,
    Elasticity_Index REAL NOT NULL
);

-- Additional tables for multi-agent system

-- Products table
CREATE TABLE IF NOT EXISTS products (
    Product_ID INTEGER PRIMARY KEY,
    Product_Name TEXT NOT NULL,
    Category TEXT NOT NULL,
    Subcategory TEXT,
    Unit_Cost REAL NOT NULL,
    Supplier_ID INTEGER,
    Pack_Size INTEGER DEFAULT 1,
    Case_Size INTEGER DEFAULT 1,
    Min_Order_Quantity INTEGER DEFAULT 1,
    Is_Perishable BOOLEAN DEFAULT 0,
    Shelf_Life_Days INTEGER,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stores table
CREATE TABLE IF NOT EXISTS stores (
    Store_ID INTEGER PRIMARY KEY,
    Store_Name TEXT NOT NULL,
    Location TEXT NOT NULL,
    Size_Sqft INTEGER,
    Region TEXT,
    Store_Type TEXT,
    Open_Date TEXT,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Suppliers table
CREATE TABLE IF NOT EXISTS suppliers (
    Supplier_ID INTEGER PRIMARY KEY,
    Supplier_Name TEXT NOT NULL,
    Contact_Person TEXT,
    Email TEXT,
    Phone TEXT,
    Address TEXT,
    Reliability REAL DEFAULT 0.9,
    Avg_Lead_Time INTEGER,
    Min_Order_Value REAL,
    Payment_Terms TEXT,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    Order_ID TEXT PRIMARY KEY,
    Product_ID INTEGER NOT NULL,
    Store_ID INTEGER NOT NULL,
    Supplier_ID INTEGER NOT NULL,
    Quantity INTEGER NOT NULL,
    Unit_Cost REAL NOT NULL,
    Total_Cost REAL NOT NULL,
    Is_Expedited BOOLEAN DEFAULT 0,
    Order_Date TIMESTAMP NOT NULL,
    Expected_Delivery_Date TIMESTAMP,
    Actual_Delivery_Date TIMESTAMP,
    Status TEXT NOT NULL,
    Consolidated_ID TEXT,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Product_ID) REFERENCES products(Product_ID),
    FOREIGN KEY (Store_ID) REFERENCES stores(Store_ID),
    FOREIGN KEY (Supplier_ID) REFERENCES suppliers(Supplier_ID)
);

-- Consolidated orders table
CREATE TABLE IF NOT EXISTS consolidated_orders (
    Consolidated_ID TEXT PRIMARY KEY,
    Supplier_ID INTEGER NOT NULL,
    Store_ID INTEGER NOT NULL,
    Order_Count INTEGER NOT NULL,
    Total_Cost REAL NOT NULL,
    Status TEXT NOT NULL,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Supplier_ID) REFERENCES suppliers(Supplier_ID),
    FOREIGN KEY (Store_ID) REFERENCES stores(Store_ID)
);

-- Stock transfers table
CREATE TABLE IF NOT EXISTS stock_transfers (
    Transfer_ID TEXT PRIMARY KEY,
    Product_ID INTEGER NOT NULL,
    Source_Store_ID INTEGER NOT NULL,
    Destination_Store_ID INTEGER NOT NULL,
    Quantity INTEGER NOT NULL,
    Transfer_Date TIMESTAMP,
    Status TEXT NOT NULL,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Product_ID) REFERENCES products(Product_ID),
    FOREIGN KEY (Source_Store_ID) REFERENCES stores(Store_ID),
    FOREIGN KEY (Destination_Store_ID) REFERENCES stores(Store_ID)
);

-- Price history table
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Product_ID INTEGER NOT NULL,
    Store_ID INTEGER NOT NULL,
    Old_Price REAL NOT NULL,
    New_Price REAL NOT NULL,
    Change_Date TIMESTAMP NOT NULL,
    Change_Reason TEXT,
    Created_By TEXT,
    FOREIGN KEY (Product_ID) REFERENCES products(Product_ID),
    FOREIGN KEY (Store_ID) REFERENCES stores(Store_ID)
);

-- Sales transactions table
CREATE TABLE IF NOT EXISTS sales_transactions (
    Transaction_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Store_ID INTEGER NOT NULL,
    Transaction_Date TIMESTAMP NOT NULL,
    Customer_ID INTEGER,
    Total_Amount REAL NOT NULL,
    Payment_Method TEXT,
    FOREIGN KEY (Store_ID) REFERENCES stores(Store_ID)
);

-- Sales items table
CREATE TABLE IF NOT EXISTS sales_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Transaction_ID INTEGER NOT NULL,
    Product_ID INTEGER NOT NULL,
    Quantity INTEGER NOT NULL,
    Unit_Price REAL NOT NULL,
    Discount REAL DEFAULT 0,
    FOREIGN KEY (Transaction_ID) REFERENCES sales_transactions(Transaction_ID),
    FOREIGN KEY (Product_ID) REFERENCES products(Product_ID)
);

-- Customer segments table
CREATE TABLE IF NOT EXISTS customer_segments (
    Segment_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Segment_Name TEXT NOT NULL,
    Description TEXT,
    Size INTEGER,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    Customer_ID INTEGER PRIMARY KEY,
    Segment_ID INTEGER,
    First_Purchase_Date TIMESTAMP,
    Last_Purchase_Date TIMESTAMP,
    Total_Purchases INTEGER DEFAULT 0,
    Total_Spend REAL DEFAULT 0,
    Avg_Order_Value REAL DEFAULT 0,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Segment_ID) REFERENCES customer_segments(Segment_ID)
);

-- Agent actions log
CREATE TABLE IF NOT EXISTS agent_actions (
    Action_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Agent_Type TEXT NOT NULL,
    Action_Type TEXT NOT NULL,
    Related_ID TEXT,  -- Could be product_id, store_id, order_id, etc.
    Description TEXT,
    Status TEXT,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent recommendations
CREATE TABLE IF NOT EXISTS agent_recommendations (
    Recommendation_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Agent_Type TEXT NOT NULL,
    Recommendation_Type TEXT NOT NULL,
    Product_ID INTEGER,
    Store_ID INTEGER,
    Description TEXT NOT NULL,
    Justification TEXT,
    Status TEXT DEFAULT 'pending',
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Product_ID) REFERENCES products(Product_ID),
    FOREIGN KEY (Store_ID) REFERENCES stores(Store_ID)
);

-- System alerts
CREATE TABLE IF NOT EXISTS system_alerts (
    Alert_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Alert_Type TEXT NOT NULL,
    Severity TEXT NOT NULL,
    Related_ID TEXT,  -- Could be product_id, store_id, order_id, etc.
    Description TEXT NOT NULL,
    Status TEXT DEFAULT 'active',
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Resolved_At TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_demand_product_store ON demand_forecasting(Product_ID, Store_ID);
CREATE INDEX IF NOT EXISTS idx_demand_date ON demand_forecasting(Date);
CREATE INDEX IF NOT EXISTS idx_inventory_product_store ON inventory_monitoring(Product_ID, Store_ID);
CREATE INDEX IF NOT EXISTS idx_pricing_product_store ON pricing_optimization(Product_ID, Store_ID);
CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(Product_ID);
CREATE INDEX IF NOT EXISTS idx_orders_store ON orders(Store_ID);
CREATE INDEX IF NOT EXISTS idx_orders_supplier ON orders(Supplier_ID);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(Status);
CREATE INDEX IF NOT EXISTS idx_sales_transactions_store ON sales_transactions(Store_ID);
CREATE INDEX IF NOT EXISTS idx_sales_transactions_date ON sales_transactions(Transaction_Date);
CREATE INDEX IF NOT EXISTS idx_sales_items_transaction ON sales_items(Transaction_ID);
CREATE INDEX IF NOT EXISTS idx_sales_items_product ON sales_items(Product_ID);
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON agent_recommendations(Status);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON system_alerts(Status);