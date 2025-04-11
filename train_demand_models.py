import os
import sys
import logging
import traceback

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import necessary modules
from agents.demand_agent import DemandForecastingAgent
import sqlite3
import pandas as pd

def train_demand_models():
    """
    Comprehensive script to train demand forecasting models with detailed logging
    """
    # Database path
    db_path = 'data/retail_inventory.db'
    
    # Configuration for model training
    config = {
        "models_dir": "models/demand",
        "llm_model": "llama3"
    }
    
    # Ensure models directory exists
    os.makedirs(config["models_dir"], exist_ok=True)
    
    # Ollama base URL (adjust as needed)
    ollama_base_url = "http://localhost:11434"
    
    # Create demand forecasting agent
    demand_agent = DemandForecastingAgent(config, db_path, ollama_base_url)
    
    # Get all unique categories
    conn = sqlite3.connect(db_path)
    
    try:
        # Detailed query to get categories with data
        query = """
        SELECT DISTINCT p.Category 
        FROM products p
        JOIN demand_forecasting df ON p.Product_ID = df.Product_ID
        WHERE p.Active = 1
        """
        
        # Use pandas for more detailed data exploration
        categories_df = pd.read_sql_query(query, conn)
        print("Categories found with demand data:")
        print(categories_df)
        
        categories = categories_df['Category'].tolist()
        print("\nCategories list:", categories)
        
        # Manually inspect a sample of the data for each category
        for category in categories:
            print(f"\nInspecting category: {category}")
            sample_query = f"""
            SELECT * 
            FROM demand_forecasting df
            JOIN products p ON df.Product_ID = p.Product_ID
            WHERE p.Category = '{category}'
            LIMIT 5
            """
            sample_df = pd.read_sql_query(sample_query, conn)
            print(sample_df)
            print("\nSample columns:", list(sample_df.columns))
    
    except Exception as e:
        print(f"Error exploring data: {e}")
        traceback.print_exc()
    finally:
        conn.close()
    
    # Attempt to train models
    try:
        # Train models with force retrain
        training_results = demand_agent.train_models(force_retrain=True)
        
        # Log detailed results
        print("\nModel Training Results:")
        print(f"Models Trained: {training_results.get('models_trained', 0)}")
        print(f"Categories Attempted: {training_results.get('categories_attempted', 0)}")
        print(f"Full Results: {training_results}")
    
    except Exception as e:
        print(f"Error during model training: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        demand_agent.cleanup()

if __name__ == "__main__":
    train_demand_models()