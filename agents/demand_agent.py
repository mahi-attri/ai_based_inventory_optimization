import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
import requests
import json
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle
import os
import re

try:
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

class DemandForecastingAgent:
    """
    Agent responsible for predicting future product demand based on 
    historical sales data and contextual factors.
    """
    
    def __init__(self, config: Dict, db_path: str, ollama_base_url: str):
        """
        Initialize the Demand Forecasting Agent
        
        Args:
            config: Configuration dictionary for agent settings
            db_path: Path to SQLite database 
            ollama_base_url: URL for Ollama API
        """
        self.config = config
        self.db_conn = sqlite3.connect(db_path)
        self.ollama_url = ollama_base_url
        self.llm_model = config.get("llm_model", "llama3")
        self.message_bus = None  # Will be set by orchestrator
        
        # Initialize logging
        self.logger = logging.getLogger("demand_agent")
        
        # Initialize ML models 
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        
        # Load pre-trained models if they exist
        self.models_dir = config.get("models_dir", "models/demand")
        
        # Validate database schema
        self._validate_database_schema()
        
        # Load pre-trained models 
        self._load_models()
     
    def _validate_database_schema(self):
        """
        Validate the database schema for demand forecasting
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Get column information
            cursor.execute("PRAGMA table_info(demand_forecasting)") 
            columns = cursor.fetchall()
            
            # Log columns
            self.logger.info("Columns in demand_forecasting table:")
            column_names = [column[1] for column in columns]
            for col in column_names:
                self.logger.info(f"- {col}")
            
            # Check for required columns 
            required_columns = [
                'Product_ID', 'Store_ID', 'Date', 'Sales_Quantity',
                'Price', 'Promotions', 'Seasonality_Factors',
                'External_Factors', 'Demand_Trend', 'Customer_Segments'
            ]
            
            missing_columns = [col for col in required_columns if col not in column_names]
            
            if missing_columns:
                self.logger.warning(f"Missing columns: {missing_columns}")
                
            return len(missing_columns) == 0
        except Exception as e:
            self.logger.error(f"Error validating database schema: {e}")
            return False
     
    def connect_to_message_bus(self, message_bus):  
        """Connect to the inter-agent communication bus"""
        self.message_bus = message_bus

        try:
            # Subscribe to relevant topics
            self.message_bus.subscribe("pricing_updates", self.handle_pricing_update)
            self.message_bus.subscribe("inventory_updates", self.handle_inventory_update)
            self.message_bus.subscribe("external_events", self.handle_external_event)
            
            # Register request handler
            def safe_forecast_handler(params):
                try:
                    return self.handle_forecast_request(params)
                except Exception as e:
                    self.logger.error(f"Error in forecast request handler: {e}")
                    return {"error": str(e)}
            
            self.message_bus.register_request_handler(
                "demand_agent", 
                "get_forecast", 
                safe_forecast_handler
            )
            
            self.logger.info("Successfully registered demand agent message bus handlers")
         
        except Exception as e:
            self.logger.error(f"Error registering message bus handlers: {e}")
        
    def _load_models(self):
        """
        Load pre-trained ML models if they exist
        """
        # Ensure models directory exists 
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            self.logger.info(f"Created models directory: {self.models_dir}")
            return
             
        # Try to load product category models
        try:
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
            
            for model_file in model_files:
                try:
                    category = model_file.split('_')[0]
                    model_path = os.path.join(self.models_dir, model_file)
                    
                    with open(model_path, 'rb') as f:
                        if model_file.endswith('_model.pkl'):
                            self.models[category] = pickle.load(f)
                        elif model_file.endswith('_encoder.pkl'):
                            self.encoders[category] = pickle.load(f)
                        elif model_file.endswith('_scaler.pkl'):
                            self.scalers[category] = pickle.load(f)
                            
                    self.logger.info(f"Loaded {model_file}")
                except Exception as e:
                    self.logger.error(f"Error loading {model_file}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in _load_models: {e}")
    
    def handle_forecast_request(self, params: Dict) -> Dict:
        """
        Handle forecast request from other agents
        
        Args:
            params: Request parameters
            
        Returns:
            Forecast data
        """
        try:
            # Validate input parameters
            product_id = params.get("product_id")
            store_id = params.get("store_id")
            horizon_days = params.get("horizon_days", 14)
            
            # Comprehensive input validation
            if not product_id:
                return {"error": "Missing product_id"}
            if not store_id:
                return {"error": "Missing store_id"}
            
            # Ensure horizon_days is within reasonable bounds
            horizon_days = max(1, min(horizon_days, 30))
            
            # Generate forecast with extended error handling
            try:
                forecast = self.generate_forecast(product_id, store_id, horizon_days)
                
                # Additional validation of forecast
                if not forecast or "error" in forecast:
                    return {
                        "error": "Failed to generate forecast",
                        "details": forecast.get("error", "Unknown error")
                    }
                
                return forecast
            
            except Exception as e:
                self.logger.error(f"Comprehensive forecast generation error: {e}")
                return {
                    "error": "Unexpected error in forecast generation",
                    "details": str(e)
                }
        
        except Exception as e:
            self.logger.error(f"Critical error in forecast request handling: {e}")
            return {"error": "Critical system error"}
            
    def handle_pricing_update(self, message: Dict):
        """
        Handle updates from the Pricing Optimization Agent
        
        Args:
            message: Message containing pricing update information
        """
        try:
            product_id = message.get("product_id")
            store_id = message.get("store_id") 
            price_change = message.get("price_change", 0)
            
            if not product_id or not store_id:
                self.logger.warning("Incomplete pricing update message")
                return
            
            # Log the pricing update
            self.logger.info(f"Received pricing update for product {product_id}, store {store_id}: price change {price_change}")
            
            # If significant price change, evaluate impact  
            if abs(price_change) > 0.05:  # 5% price change
                self.evaluate_price_impact(product_id, store_id, price_change)
        
        except Exception as e:
            self.logger.error(f"Error in handle_pricing_update: {e}") 

    def handle_inventory_update(self, message: Dict):
        """
        Handle updates from the Inventory Management Agent
        
        Args:
            message: Message containing inventory update information
        """
        try:
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            
            if not product_id or not store_id:
                self.logger.warning("Incomplete inventory update message")
                return
            
            # Log the inventory update 
            self.logger.info(f"Received inventory update for product {product_id}, store {store_id}")
        
        except Exception as e:
            self.logger.error(f"Error in handle_inventory_update: {e}")

    def handle_external_event(self, message: Dict):
        """
        Handle external event updates
        
        Args:
            message: Message containing external event information
        """
        try:
            event_type = message.get("type")
            
            if not event_type:
                self.logger.warning("Received external event with no type")
                return
            
            # Log the external event
            self.logger.info(f"Received external event: {event_type}")
            
            # Basic event type handling
            if event_type == "weather":
                self.handle_weather_event(message)
            elif event_type == "holiday":  
                self.handle_holiday_event(message)
            elif event_type == "competitor":
                self.handle_competitor_event(message)
            else:
                self.logger.warning(f"Unhandled external event type: {event_type}")
        
        except Exception as e:
            self.logger.error(f"Error in handle_external_event: {e}")
            
    def handle_weather_event(self, message: Dict):
        try:
            self.logger.info(f"Processing weather event: {message}")
        except Exception as e:  
            self.logger.error(f"Error in handle_weather_event: {e}")
            
    def handle_holiday_event(self, message: Dict):
        try:
            self.logger.info(f"Processing holiday event: {message}")
        except Exception as e:
            self.logger.error(f"Error in handle_holiday_event: {e}")

    def handle_competitor_event(self, message: Dict):
        try:  
            self.logger.info(f"Processing competitor event: {message}")
        except Exception as e:
            self.logger.error(f"Error in handle_competitor_event: {e}")

    def evaluate_price_impact(self, product_id: int, store_id: int, price_change: float):
        """
        Evaluate the impact of a price change on demand
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            price_change: Percentage price change
        """
        try:
            # Get pricing data
            query = """
            SELECT Elasticity_Index 
            FROM pricing_optimization
            WHERE Product_ID = ? AND Store_ID = ?
            """
            
            cursor = self.db_conn.cursor()
            cursor.execute(query, (product_id, store_id))
            result = cursor.fetchone()
            
            if not result:
                self.logger.warning(f"No elasticity data for product {product_id}, store {store_id}")
                return
            
            elasticity = result[0]
            
            # Calculate expected demand impact
            expected_demand_change = -1 * elasticity * price_change
            
            self.logger.info(f"Price impact analysis for product {product_id}, store {store_id}:")
            self.logger.info(f"- Price change: {price_change}%") 
            self.logger.info(f"- Elasticity: {elasticity}")
            self.logger.info(f"- Expected demand change: {expected_demand_change}%")
            
            # Publish demand change alert if significant
            if abs(expected_demand_change) > 10:  # 10% demand change 
                self.publish_demand_change_alert(product_id, store_id, expected_demand_change)
        
        except Exception as e:
            self.logger.error(f"Error in evaluate_price_impact: {e}")

    def publish_demand_change_alert(self, product_id: int, store_id: int, expected_change: float):
        """  
        Publish a demand change alert
        
        Args:
            product_id: The product identifier
            store_id: The store identifier 
            expected_change: Expected percentage change in demand
        """
        try:
            if not self.message_bus:
                self.logger.warning("Message bus not available for publishing demand change alert")
                return
            
            alert_type = "demand_spike" if expected_change > 0 else "demand_drop"
            
            message = {
                "type": alert_type,
                "product_id": product_id,
                "store_id": store_id,
                "magnitude": abs(expected_change),
                "expected_duration": "short_term",
                "confidence": min(90, 50 + abs(expected_change)),
                "source": "price_change",  
                "timestamp": datetime.now().isoformat()
            }
            
            self.message_bus.publish("demand_alerts", message) 
            self.logger.info(f"Published demand change alert: {message}")
        
        except Exception as e:
            self.logger.error(f"Error in publish_demand_change_alert: {e}")

    def generate_forecast(self, product_id: int, store_id: int, horizon_days: int) -> Dict:
        """
        Generate demand forecast for a given product and store with fallback mechanism
        
        Args:
            product_id: The product ID
            store_id: The store ID 
            horizon_days: Number of days to forecast
        
        Returns:
            Dict with forecasted demands
        """
        try:
            # Get product info
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT Category FROM products WHERE Product_ID = ?", (product_id,))
            category_result = cursor.fetchone()
            
            if not category_result:
                self.logger.warning(f"No product found with ID: {product_id}")
                return {"error": f"Product not found: {product_id}"}
                
            category = category_result[0]
            
            # Check if model exists for this category
            if category not in self.models:
                # Fallback: Try to use a global model or create a simple default model
                self.logger.warning(f"No forecasting model found for category: {category}")
                
                # Option 1: Use a generic global model if available
                if "global" in self.models:
                    category = "global"
                else:
                    # Option 2: Create a simple default forecast
                    return self._generate_default_forecast(product_id, store_id, horizon_days)
            
            # Query historical data
            query = """
            SELECT Date, Sales_Quantity, Price, Promotions, Seasonality_Factors, External_Factors 
            FROM demand_forecasting
            WHERE Product_ID = ? AND Store_ID = ?
            ORDER BY Date DESC
            LIMIT 180
            """
            
            df = pd.read_sql_query(query, self.db_conn, params=(product_id, store_id))
            
            if df.empty:
                return {"error": "No historical data found"}
            
            # Prepare features
            X = df[['Price', 'Promotions', 'Seasonality_Factors', 'External_Factors']]
            y = df['Sales_Quantity']
            
            # Scale features
            X_scaled = self.scalers[category].transform(X)
            
            # Make predictions
            model = self.models[category]
            y_pred = model.predict(X_scaled[-horizon_days:])
            
            # Format forecast
            dates = pd.date_range(start=df['Date'].max(), periods=horizon_days+1)[1:] 
            forecast = {
                'product_id': product_id,
                'store_id': store_id,
                'forecast': [{
                    'date': date.isoformat(),
                    'demand': int(demand)
                } for date, demand in zip(dates, y_pred)]
            }
            
            self.logger.info(f"Generated forecast for product {product_id}, store {store_id}: {forecast}")
            
            return forecast
        
        except Exception as e:
            self.logger.error(f"Error in generate_forecast: {e}")
            return {"error": str(e)}

    def _generate_default_forecast(self, product_id: int, store_id: int, horizon_days: int) -> Dict:
        """
        Generate a default forecast when no specific model is available
        
        Args:
            product_id: The product ID
            store_id: The store ID 
            horizon_days: Number of days to forecast
        
        Returns:
            Dict with default forecasted demands
        """
        try:
            # Query recent historical average
            query = """
            SELECT AVG(Sales_Quantity) as avg_sales
            FROM demand_forecasting
            WHERE Product_ID = ? AND Store_ID = ?
            AND Date > date('now', '-30 days')
            """
            
            cursor = self.db_conn.cursor()
            cursor.execute(query, (product_id, store_id))
            result = cursor.fetchone()
            
            # Default to a baseline if no historical data
            avg_sales = result[0] if result and result[0] is not None else 10
            
            # Generate default forecast
            dates = pd.date_range(start=datetime.now(), periods=horizon_days+1)[1:]
            forecast = {
                'product_id': product_id,
                'store_id': store_id,
                'forecast_type': 'default',
                'forecast': [{
                    'date': date.isoformat(),
                    'demand': int(max(1, avg_sales + np.random.normal(0, avg_sales * 0.1)))
                } for date in dates]
            }
            
            self.logger.info(f"Generated default forecast for product {product_id}, store {store_id}")
            
            return forecast
        
        except Exception as e:
            self.logger.error(f"Error in _generate_default_forecast: {e}")
            return {
                "error": "Unable to generate default forecast",
                "details": str(e)
            }
    
    def run_periodic_demand_forecast(self):
        """
        Run periodic demand forecasting for all products and stores
        
        Returns:
            Dict with forecast results
        """
        try:
            self.logger.info("Running periodic demand forecasting")
            
            # Get all product/store combinations
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT DISTINCT d.Product_ID, d.Store_ID 
                FROM demand_forecasting d
                JOIN products p ON d.Product_ID = p.Product_ID
                WHERE p.Active = 1
                LIMIT 20
            """)
            
            product_store_pairs = cursor.fetchall()
            
            if not product_store_pairs:
                self.logger.warning("No active product/store combinations found")
                return {"forecasts_generated": 0}
            
            # Generate forecasts for each product/store
            forecasts = []
            for product_id, store_id in product_store_pairs:
                try:
                    forecast = self.generate_forecast(product_id, store_id, 14)  # 14-day forecast
                    
                    if "error" not in forecast:
                        forecasts.append(forecast)
                    else:
                        self.logger.warning(f"Error forecasting product {product_id}, store {store_id}: {forecast['error']}")
                except Exception as e:
                    self.logger.error(f"Error forecasting product {product_id}, store {store_id}: {e}")
            
            # Summarize and log results
            self.logger.info(f"Generated {len(forecasts)} forecasts out of {len(product_store_pairs)} product/store pairs")
            
            # Publish forecast summaries to message bus
            if self.message_bus and forecasts:
                self.publish_forecast_summaries(forecasts)
            
            return {
                "forecasts_generated": len(forecasts),
                "attempted": len(product_store_pairs),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Error in run_periodic_demand_forecast: {e}")
            return {"error": str(e), "forecasts_generated": 0}
    
    def publish_forecast_summaries(self, forecasts: List[Dict]):
        """
        Publish forecast summaries to the message bus
        
        Args:
            forecasts: List of forecast dictionaries
        """
        try:
            # Create summary message
            summary = {
                "type": "demand_forecast_summary",
                "timestamp": datetime.now().isoformat(),
                "forecast_count": len(forecasts),
                "summaries": []
            }
            
            # Add summary for each forecast
            for forecast in forecasts:
                product_id = forecast['product_id']
                store_id = forecast['store_id']
                forecast_data = forecast['forecast']
                
                # Calculate total and average demand
                total_demand = sum(item['demand'] for item in forecast_data)
                avg_demand = total_demand / len(forecast_data) if forecast_data else 0
                
                # Add to summaries
                summary["summaries"].append({
                    "product_id": product_id,
                    "store_id": store_id,
                    "days": len(forecast_data),
                    "total_demand": total_demand,
                    "avg_daily_demand": avg_demand
                })
            
            # Publish to message bus
            self.message_bus.publish("demand_forecasts", summary)
            self.logger.info(f"Published forecast summaries for {len(forecasts)} products")
        
        except Exception as e:
            self.logger.error(f"Error in publish_forecast_summaries: {e}")

    def train_models(self, force_retrain=False):
        """
        Train or retrain demand forecasting models
        
        Args:
            force_retrain: Force retraining even if models exist
            
        Returns:
            Dict with training results
        """
        try:
            self.logger.info("Starting model training")
            
            # Get product categories
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT DISTINCT Category FROM products WHERE Active = 1")
            categories = [row[0] for row in cursor.fetchall()]
            
            if not categories:
                self.logger.warning("No active product categories found")
                return {"models_trained": 0}
            
            # Initialize trained_models before the loop
            trained_models = 0
            
            # Train a model for each category
            for category in categories:
                try:
                    # Check if model exists and skip if not forcing retrain
                    model_path = os.path.join(self.models_dir, f"{category}_model.pkl")
                    if os.path.exists(model_path) and not force_retrain:
                        self.logger.info(f"Skipping training for category {category} (model exists)")
                        continue
                    
                    # Get training data
                    query = """
                    SELECT df.* 
                    FROM demand_forecasting df
                    JOIN products p ON df.Product_ID = p.Product_ID
                    WHERE p.Category = ?
                    ORDER BY df.Date
                    """
                    
                    df = pd.read_sql_query(query, self.db_conn, params=(category,))
                    
                    if df.empty:
                        self.logger.warning(f"No training data for category {category}")
                        continue
                    
                    # Prepare features and target
                    X = df[['Price', 'Promotions', 'Seasonality_Factors', 'External_Factors']]
                    y = df['Sales_Quantity']
                    
                    # Create and fit scaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Create and train model
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                    model.fit(X_scaled, y)
                    
                    # Save model and scaler
                    os.makedirs(self.models_dir, exist_ok=True)
                    
                    with open(os.path.join(self.models_dir, f"{category}_model.pkl"), 'wb') as f:
                        pickle.dump(model, f)
                    
                    with open(os.path.join(self.models_dir, f"{category}_scaler.pkl"), 'wb') as f:
                        pickle.dump(scaler, f)
                    
                    # Add to models and scalers
                    self.models[category] = model
                    self.scalers[category] = scaler
                    
                    trained_models += 1
                    self.logger.info(f"Trained model for category {category}")
                    
                except Exception as e:
                    self.logger.error(f"Error training model for category {category}: {e}")
            
            # Add global fallback model if no models were trained
            if trained_models == 0:
                try:
                    # Query all available data for training a global model
                    query = """
                    SELECT * 
                    FROM demand_forecasting
                    ORDER BY Date
                    """
                    
                    df = pd.read_sql_query(query, self.db_conn)
                    
                    if not df.empty:
                        # Prepare features and target
                        X = df[['Price', 'Promotions', 'Seasonality_Factors', 'External_Factors']]
                        y = df['Sales_Quantity']
                        
                        # Create and fit scaler
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Create and train global model
                        global_model = RandomForestRegressor(
                            n_estimators=100,
                            max_depth=10,
                            random_state=42
                        )
                        global_model.fit(X_scaled, y)
                        
                        # Save global model and scaler
                        os.makedirs(self.models_dir, exist_ok=True)
                        
                        with open(os.path.join(self.models_dir, "global_model.pkl"), 'wb') as f:
                            pickle.dump(global_model, f)
                        
                        with open(os.path.join(self.models_dir, "global_scaler.pkl"), 'wb') as f:
                            pickle.dump(scaler, f)
                        
                        # Add to models and scalers
                        self.models["global"] = global_model
                        self.scalers["global"] = scaler
                        
                        trained_models += 1
                        self.logger.info("Trained global fallback model")
                except Exception as e:
                    self.logger.error(f"Error training global model: {e}")
            
            self.logger.info(f"Trained {trained_models} models out of {len(categories)} categories")
            
            return {
                "models_trained": trained_models,
                "categories_attempted": len(categories),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Error in train_models: {e}")
            return {"error": str(e), "models_trained": 0}
    def cleanup(self):
        """
        Perform cleanup operations when the agent is shutting down
        """ 
        try:
            # Close database connection
            if self.db_conn: 
                self.db_conn.close()
            
            # Log cleanup
            self.logger.info("Demand Forecasting Agent cleanup completed")
        except Exception as e:  
            self.logger.error(f"Error during cleanup: {e}")