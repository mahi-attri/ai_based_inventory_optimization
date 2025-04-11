import pandas as pd
import sqlite3
import os

def import_csv_data():
    # Connect to the database
    conn = sqlite3.connect('data/retail_inventory.db')
    
    # Import demand forecasting data
    demand_path = r'C:\Users\mahia\OneDrive\Documents\retail-inventory-optimizer\demand_forecasting.csv'
    if os.path.exists(demand_path):
        df = pd.read_csv(demand_path)
        df.to_sql('demand_forecasting', conn, if_exists='append', index=False)
        print(f"Imported {len(df)} rows into demand_forecasting")
    else:
        print(f"File not found: {demand_path}")
    
    # Import inventory monitoring data
    inventory_path = r'C:\Users\mahia\OneDrive\Documents\retail-inventory-optimizer\inventory_monitoring.csv'
    if os.path.exists(inventory_path):
        df = pd.read_csv(inventory_path)
        df.to_sql('inventory_monitoring', conn, if_exists='append', index=False)
        print(f"Imported {len(df)} rows into inventory_monitoring")
    else:
        print(f"File not found: {inventory_path}")
    
    # Import pricing optimization data
    pricing_path = r'C:\Users\mahia\OneDrive\Documents\retail-inventory-optimizer\pricing_optimization.csv'
    if os.path.exists(pricing_path):
        df = pd.read_csv(pricing_path)
        df.to_sql('pricing_optimization', conn, if_exists='append', index=False)
        print(f"Imported {len(df)} rows into pricing_optimization")
    else:
        print(f"File not found: {pricing_path}")
    
    conn.close()
    print("Data import completed")

if __name__ == "__main__":
    import_csv_data()