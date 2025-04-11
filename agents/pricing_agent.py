# pricing_agent.py
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

class PricingOptimizationAgent:
    """
    Agent responsible for optimizing product pricing based on
    market conditions, inventory status, and customer behavior.
    """
    
    def __init__(self, config: Dict, db_path: str, ollama_base_url: str):
        """
        Initialize the Pricing Optimization Agent
        
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
        self.logger = logging.getLogger("pricing_agent")
        
        # Initialize ML models
        self.models = {}
        self.scalers = {}
        
        # Load pre-trained models if they exist
        self.models_dir = config.get("models_dir", "models/pricing")
        self._load_models()
        
        # Configure pricing strategy
        self.min_margin = config.get("min_margin", 0.1)  # Minimum profit margin (10%)
        self.max_discount = config.get("max_discount", 0.3)  # Maximum discount (30%)
        self.competitor_threshold = config.get("competitor_threshold", 0.05)  # 5% threshold for competitor price matching
        
    def connect_to_message_bus(self, message_bus):
        """Connect to the inter-agent communication bus"""
        self.message_bus = message_bus
        # Register request handlers
        self.message_bus.register_request_handler(
            "pricing_agent", "get_pricing_data", self.handle_pricing_data_request
        )
        
        # Subscribe to relevant topics
        self.message_bus.subscribe("demand_updates", self.handle_demand_update)
        self.message_bus.subscribe("inventory_updates", self.handle_inventory_update)
        self.message_bus.subscribe("competitor_updates", self.handle_competitor_update)
        
    def _load_models(self):
        """Load pre-trained ML models if they exist"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            self.logger.info(f"Created models directory: {self.models_dir}")
            return
            
        # Try to load product category models
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        for model_file in model_files:
            try:
                category = model_file.split('_')[0]
                model_path = os.path.join(self.models_dir, model_file)
                
                with open(model_path, 'rb') as f:
                    if model_file.endswith('_model.pkl'):
                        self.models[category] = pickle.load(f)
                    elif model_file.endswith('_scaler.pkl'):
                        self.scalers[category] = pickle.load(f)
                        
                self.logger.info(f"Loaded {model_file}")
            except Exception as e:
                self.logger.error(f"Error loading {model_file}: {e}")
    
    def _save_model(self, category: str, model_type: str, model_obj):
        """
        Save a model to disk
        
        Args:
            category: Product category
            model_type: Type of model (model, scaler)
            model_obj: The model object to save
        """
        os.makedirs(self.models_dir, exist_ok=True)
        
        file_path = os.path.join(self.models_dir, f"{category}_{model_type}.pkl")
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_obj, f)
            
        self.logger.info(f"Saved {category}_{model_type}.pkl")
    
    def get_product_category(self, product_id: int) -> str:
        """
        Get the category for a product
        
        Args:
            product_id: The product identifier
            
        Returns:
            Product category string
        """
        # In a real implementation, this would query a product catalog
        # For this demo, we'll use a simple mapping based on product ID ranges
        if product_id < 2000:
            return "electronics"
        elif product_id < 4000:
            return "clothing"
        elif product_id < 6000:
            return "groceries"
        elif product_id < 8000:
            return "home"
        else:
            return "other"
    
    def get_current_pricing_data(self, product_id: int, store_id: int) -> Dict:
        """
        Get current pricing data for a product at a store
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Dictionary with pricing details
        """
        query = """
        SELECT 
            Price, 
            Competitor_Prices, 
            Discounts, 
            Sales_Volume, 
            Customer_Reviews,
            Return_Rate, 
            Storage_Cost, 
            Elasticity_Index
        FROM 
            pricing_optimization
        WHERE 
            Product_ID = ? AND Store_ID = ?
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (product_id, store_id))
        result = cursor.fetchone()
        
        if not result:
            return None
            
        pricing_data = {
            "product_id": product_id,
            "store_id": store_id,
            "current_price": result[0],
            "competitor_price": result[1],
            "current_discount": result[2],
            "sales_volume": result[3],
            "customer_reviews": result[4],
            "return_rate": result[5],
            "storage_cost": result[6],
            "elasticity": result[7],
            "timestamp": datetime.now().isoformat()
        }
        
        return pricing_data
    
    def get_price_history(self, product_id: int, store_id: int, days: int = 90) -> pd.DataFrame:
        """
        Get price history for a product at a store
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with price history
        """
        # For this sample, we'll simulate price history 
        # In a real implementation, this would query a price history table
        
        # Get current pricing
        current_pricing = self.get_current_pricing_data(product_id, store_id)
        
        if not current_pricing:
            return pd.DataFrame()
            
        # Create synthetic price history
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        
        # Base price with some random variations
        current_price = current_pricing["current_price"]
        np.random.seed(product_id + store_id)  # For reproducible random variations
        
        # Create price variations within +/- 10% of current price
        price_variations = np.random.uniform(0.9, 1.1, size=days) * current_price
        
        # Create some discount events
        discount_events = np.zeros(days)
        for i in range(days // 15):  # Create a discount event roughly every 15 days
            event_start = np.random.randint(0, days - 7)
            event_duration = np.random.randint(3, 8)
            discount_pct = np.random.uniform(0.05, 0.2)
            
            for j in range(event_duration):
                if event_start + j < days:
                    discount_events[event_start + j] = discount_pct
        
        # Apply discounts to prices
        discounted_prices = price_variations * (1 - discount_events)
        
        # Create synthetic competitor prices
        competitor_variations = np.random.uniform(0.85, 1.15, size=days) * current_price
        
        # Create DataFrame
        price_history = pd.DataFrame({
            'Date': dates,
            'Price': discounted_prices,
            'Original_Price': price_variations,
            'Discount_Percentage': discount_events * 100,  # Convert to percentage
            'Competitor_Price': competitor_variations
        })
        
        # Sort by date from oldest to newest
        price_history = price_history.sort_values('Date')
        
        return price_history
    
    def get_sales_data(self, product_id: int, store_id: int, days: int = 90) -> pd.DataFrame:
        """
        Get sales data for a product at a store
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with sales data
        """
        query = """
        SELECT 
            Date, 
            Sales_Quantity, 
            Price
        FROM 
            demand_forecasting
        WHERE 
            Product_ID = ? AND 
            Store_ID = ? AND
            Date >= date('now', ?)
        ORDER BY
            Date ASC
        """
        
        days_param = f"-{days} days"
        
        try:
            df = pd.read_sql_query(
                query, 
                self.db_conn, 
                params=(product_id, store_id, days_param)
            )
            
            # Convert date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            return df
        except Exception as e:
            self.logger.error(f"Error getting sales data: {e}")
            return pd.DataFrame()
    
    def calculate_price_elasticity(self, product_id: int, store_id: int) -> float:
        """
        Calculate price elasticity of demand for a product
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Price elasticity value
        """
        # Get sales and price data
        sales_data = self.get_sales_data(product_id, store_id)
        
        if sales_data.empty or len(sales_data) < 10:
            # Not enough data, use default elasticity or lookup from database
            pricing_data = self.get_current_pricing_data(product_id, store_id)
            return pricing_data.get("elasticity", 1.0) if pricing_data else 1.0
            
        # Calculate percentage changes
        sales_data['Price_Lag'] = sales_data['Price'].shift(1)
        sales_data['Sales_Lag'] = sales_data['Sales_Quantity'].shift(1)
        
        sales_data['Price_Pct_Change'] = (sales_data['Price'] - sales_data['Price_Lag']) / sales_data['Price_Lag']
        sales_data['Sales_Pct_Change'] = (sales_data['Sales_Quantity'] - sales_data['Sales_Lag']) / sales_data['Sales_Lag']
        
        # Remove rows with invalid values
        sales_data = sales_data.dropna()
        sales_data = sales_data[sales_data['Price_Pct_Change'] != 0]  # Avoid division by zero
        
        # Calculate elasticity for each data point
        sales_data['Elasticity'] = sales_data['Sales_Pct_Change'] / sales_data['Price_Pct_Change']
        
        # Remove extreme outliers (elasticity values beyond +/-10)
        sales_data = sales_data[(sales_data['Elasticity'] > -10) & (sales_data['Elasticity'] < 10)]
        
        if sales_data.empty:
            # Fallback to default if all values were outliers
            pricing_data = self.get_current_pricing_data(product_id, store_id)
            return pricing_data.get("elasticity", 1.0) if pricing_data else 1.0
            
        # Calculate median elasticity
        median_elasticity = abs(sales_data['Elasticity'].median())
        
        # Constrain to reasonable range
        median_elasticity = min(3.0, max(0.1, median_elasticity))
        
        return median_elasticity
    
    def train_pricing_model(self, product_category: str, force_retrain: bool = False):
        """
        Train or retrain ML model for price optimization
        
        Args:
            product_category: The product category
            force_retrain: Whether to force retraining even if model exists
        """
        # Check if model already exists and we're not forcing a retrain
        if product_category in self.models and not force_retrain:
            self.logger.info(f"Model for {product_category} already exists")
            return
            
        self.logger.info(f"Training pricing model for {product_category}")
        
        # Get sample of products in this category
        products = self._get_products_in_category(product_category, limit=20)
        
        if not products:
            self.logger.warning(f"No products found for category {product_category}")
            return
            
        # Collect training data from all products
        all_data = []
        
        for product_id, store_id in products:
            # Get sales data
            sales_data = self.get_sales_data(product_id, store_id)
            
            if not sales_data.empty:
                # Get pricing data
                pricing_data = self.get_current_pricing_data(product_id, store_id)
                
                if pricing_data:
                    # Add pricing features to each sales record
                    for col in ['competitor_price', 'customer_reviews', 'return_rate', 'storage_cost', 'elasticity']:
                        sales_data[col] = pricing_data.get(col, 0)
                    
                    # Calculate price relative to competitor
                    sales_data['price_to_competitor_ratio'] = sales_data['Price'] / sales_data['competitor_price']
                    
                    all_data.append(sales_data)
        
        if not all_data:
            self.logger.warning(f"No training data found for category {product_category}")
            return
            
        # Combine all data
        training_data = pd.concat(all_data, ignore_index=True)
        
        # Prepare features and target
        features = [
            'Price', 'competitor_price', 'customer_reviews', 
            'return_rate', 'storage_cost', 'elasticity',
            'price_to_competitor_ratio'
        ]
        
        X = training_data[features].values
        y = training_data['Sales_Quantity'].values
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_scaled, y)
        
        # Save model and scaler
        self.models[product_category] = model
        self.scalers[product_category] = scaler
        
        # Save to disk
        self._save_model(product_category, "model", model)
        self._save_model(product_category, "scaler", scaler)
        
        self.logger.info(f"Trained pricing model for {product_category} with {len(training_data)} samples")
    
    def _get_products_in_category(self, category: str, limit: int = 20) -> List[Tuple[int, int]]:
        """
        Get sample products in a category
        
        Args:
            category: Product category
            limit: Maximum number of products to retrieve
            
        Returns:
            List of (product_id, store_id) tuples
        """
        # In a real implementation, this would query a product catalog
        # For this demo, we'll simulate based on product ID ranges
        
        # Define ID ranges for each category
        category_ranges = {
            "electronics": (0, 2000),
            "clothing": (2000, 4000),
            "groceries": (4000, 6000),
            "home": (6000, 8000),
            "other": (8000, 10000)
        }
        
        if category not in category_ranges:
            return []
            
        start_id, end_id = category_ranges[category]
        
        query = """
        SELECT DISTINCT Product_ID, Store_ID 
        FROM pricing_optimization
        WHERE Product_ID >= ? AND Product_ID < ?
        LIMIT ?
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (start_id, end_id, limit))
        results = cursor.fetchall()
        
        return results
    
    def predict_sales_at_price(self, product_id: int, store_id: int, price: float) -> float:
        """
        Predict sales quantity at a given price
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            price: The price to evaluate
            
        Returns:
            Predicted sales quantity
        """
        # Get product category
        category = self.get_product_category(product_id)
        
        # Ensure model exists for this category
        if category not in self.models:
            self.train_pricing_model(category)
            
            # Check if training succeeded
            if category not in self.models:
                # Fallback to elasticity-based prediction
                return self._predict_sales_using_elasticity(product_id, store_id, price)
        
        # Get pricing data
        pricing_data = self.get_current_pricing_data(product_id, store_id)
        
        if not pricing_data:
            return 0
            
        # Prepare feature vector
        features = [
            price,
            pricing_data['competitor_price'],
            pricing_data['customer_reviews'],
            pricing_data['return_rate'],
            pricing_data['storage_cost'],
            pricing_data['elasticity'],
            price / pricing_data['competitor_price']
        ]
        
        # Scale features
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scalers[category].transform(X)
        
        # Make prediction
        predicted_sales = self.models[category].predict(X_scaled)[0]
        
        # Ensure non-negative value
        return max(0, predicted_sales)
    
    def _predict_sales_using_elasticity(self, product_id: int, store_id: int, new_price: float) -> float:
        """
        Predict sales using price elasticity as fallback
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            new_price: New price to evaluate
            
        Returns:
            Predicted sales quantity
        """
        # Get current pricing and sales data
        pricing_data = self.get_current_pricing_data(product_id, store_id)
        
        if not pricing_data:
            return 0
            
        current_price = pricing_data['current_price']
        current_sales = pricing_data['sales_volume']
        elasticity = pricing_data['elasticity']
        
        if current_price == 0:
            return current_sales  # Avoid division by zero
            
        # Calculate percentage price change
        price_change_pct = (new_price - current_price) / current_price
        
        # Calculate expected sales change using elasticity
        sales_change_pct = -1 * elasticity * price_change_pct
        
        # Calculate predicted sales
        predicted_sales = current_sales * (1 + sales_change_pct)
        
        # Ensure non-negative value
        return max(0, predicted_sales)
    
    def calculate_optimal_price(self, product_id: int, store_id: int) -> Dict:
        """
        Calculate the optimal price for a product
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Price optimization results
        """
        # Get current pricing data
        pricing_data = self.get_current_pricing_data(product_id, store_id)
        
        if not pricing_data:
            return {"error": "No pricing data available"}
            
        current_price = pricing_data['current_price']
        competitor_price = pricing_data['competitor_price']
        storage_cost = pricing_data['storage_cost']
        
        # Get inventory data
        inventory_data = self._get_inventory_data(product_id, store_id)
        
        # Define price range to evaluate
        min_price = max(storage_cost * 1.1, current_price * 0.7)  # Ensure minimum markup over cost
        max_price = current_price * 1.3  # Max 30% increase
        
        # Create price points to evaluate
        price_points = np.linspace(min_price, max_price, 20)
        
        # Evaluate each price point
        results = []
        
        for price in price_points:
            # Predict sales at this price
            predicted_sales = self.predict_sales_at_price(product_id, store_id, price)
            
            # Calculate gross profit
            profit_per_unit = price - storage_cost
            total_profit = predicted_sales * profit_per_unit
            
            # Calculate inventory impact
            inventory_impact = 0
            if inventory_data:
                stock_level = inventory_data.get('stock_level', 0)
                reorder_point = inventory_data.get('reorder_point', 0)
                
                # Prefer higher sales if stock is high, lower sales if stock is low
                if stock_level > 1.5 * reorder_point:
                    inventory_impact = 0.2  # Bonus for higher sales to reduce excess stock
                elif stock_level < reorder_point:
                    inventory_impact = -0.2  # Penalty for higher sales when stock is low
            
            # Calculate competitor impact (penalty for being much higher than competitor)
            competitor_impact = 0
            if price > competitor_price * (1 + self.competitor_threshold):
                competitor_impact = -0.1 * ((price / competitor_price) - (1 + self.competitor_threshold))
            
            # Calculate total score (profit with adjustments)
            total_score = total_profit * (1 + inventory_impact + competitor_impact)
            
            results.append({
                "price": price,
                "predicted_sales": predicted_sales,
                "profit_per_unit": profit_per_unit,
                "total_profit": total_profit,
                "inventory_impact": inventory_impact,
                "competitor_impact": competitor_impact,
                "total_score": total_score
            })
        
        # Find price with highest score
        best_result = max(results, key=lambda x: x['total_score'])
        
        # Calculate price change
        price_change = (best_result['price'] - current_price) / current_price
        
        # Calculate expected sales impact
        current_sales = pricing_data['sales_volume']
        sales_change = (best_result['predicted_sales'] - current_sales) / current_sales if current_sales > 0 else 0
        
        # Enhance with LLM reasoning
        enhanced_recommendation = self._enhance_price_recommendation(
            product_id, store_id,
            pricing_data, inventory_data,
            best_result, price_change, sales_change
        )
        
        return enhanced_recommendation
    
    def _get_inventory_data(self, product_id: int, store_id: int) -> Dict:
        """
        Get inventory data for a product
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Inventory data or None if not available
        """
        query = """
        SELECT 
            Stock_Levels, 
            Reorder_Point, 
            Supplier_Lead_Time_days
        FROM 
            inventory_monitoring
        WHERE 
            Product_ID = ? AND Store_ID = ?
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (product_id, store_id))
        result = cursor.fetchone()
        
        if not result:
            return None
            
        return {
            "stock_level": result[0],
            "reorder_point": result[1],
            "lead_time": result[2]
        }
    
    def _enhance_price_recommendation(self, product_id: int, store_id: int,
                                     pricing_data: Dict, inventory_data: Dict,
                                     best_result: Dict, price_change: float,
                                     sales_change: float) -> Dict:
        """
        Enhance price recommendation with LLM reasoning
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            pricing_data: Current pricing data
            inventory_data: Current inventory data
            best_result: Best price optimization result
            price_change: Calculated price change percentage
            sales_change: Expected sales change percentage
            
        Returns:
            Enhanced price recommendation
        """
        # Format data for LLM prompt
        current_price = pricing_data['current_price']
        competitor_price = pricing_data['competitor_price']
        storage_cost = pricing_data['storage_cost']
        elasticity = pricing_data['elasticity']
        
        stock_level = inventory_data.get('stock_level', 'unknown') if inventory_data else 'unknown'
        reorder_point = inventory_data.get('reorder_point', 'unknown') if inventory_data else 'unknown'
        lead_time = inventory_data.get('lead_time', 'unknown') if inventory_data else 'unknown'
        
        best_price = best_result['price']
        predicted_sales = best_result['predicted_sales']
        total_profit = best_result['total_profit']
        
        # Get demand trend data
        demand_trend = self._get_demand_trend(product_id, store_id)
        
        # Create prompt for LLM
        prompt = f"""
        You are a pricing optimization expert for a retail chain. I need your analysis on the following pricing recommendation:
        
        Product ID: {product_id}
        Store ID: {store_id}
        
        Current situation:
        - Current price: ${current_price:.2f}
        - Competitor price: ${competitor_price:.2f}
        - Storage/cost per unit: ${storage_cost:.2f}
        - Price elasticity: {elasticity:.2f}
        - Current stock level: {stock_level} units
        - Reorder point: {reorder_point} units
        - Supplier lead time: {lead_time} days
        - Demand trend: {demand_trend}
        
        Calculated recommendation:
        - Recommended price: ${best_price:.2f}
        - Price change: {price_change*100:.1f}%
        - Predicted sales at new price: {predicted_sales:.1f} units
        - Expected sales volume change: {sales_change*100:.1f}%
        - Estimated profit per unit: ${best_result['profit_per_unit']:.2f}
        - Estimated total profit: ${total_profit:.2f}
        
        Please analyze this pricing recommendation and provide:
        1. Your assessment of the recommended price
        2. Whether any adjustments are needed considering inventory, competitors, and demand trends
        3. Any risks associated with this price change
        4. Implementation recommendations (immediate change, gradual change, etc.)
        
        Format your response as follows:
        ASSESSMENT: [brief assessment]
        
        PRICE_RECOMMENDATION: [one of: ACCEPT, ADJUST_UP, ADJUST_DOWN]
        RECOMMENDED_PRICE: [your price recommendation if different from calculated]
        
        JUSTIFICATION:
        [detailed justification]
        
        RISKS:
        - [risk 1]
        - [risk 2]
        - ...
        
        IMPLEMENTATION:
        [implementation recommendation]
        """
        
        # Call LLM
        llm_response = self._call_ollama_api(prompt)
        
        # Parse LLM response
        parsed_response = self._parse_price_recommendation_response(llm_response, best_price)
        
        # Combine all information
        final_recommendation = {
            "product_id": product_id,
            "store_id": store_id,
            "current_price": current_price,
            "competitor_price": competitor_price,
            "calculated_optimal_price": best_price,
            "recommended_price": parsed_response["recommended_price"],
            "price_change": (parsed_response["recommended_price"] - current_price) / current_price,
            "recommendation_type": parsed_response["recommendation_type"],
            "justification": parsed_response["justification"],
            "risks": parsed_response["risks"],
            "implementation": parsed_response["implementation"],
            "predicted_sales": predicted_sales,
            "expected_sales_impact": sales_change,
            "demand_trend": demand_trend,
            "timestamp": datetime.now().isoformat()
        }
        
        return final_recommendation
    
    def _get_demand_trend(self, product_id: int, store_id: int) -> str:
        """
        Get demand trend information for a product
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Demand trend description
        """
        query = """
        SELECT 
            Demand_Trend
        FROM 
            demand_forecasting
        WHERE 
            Product_ID = ? AND Store_ID = ?
        ORDER BY 
            Date DESC
        LIMIT 1
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (product_id, store_id))
        result = cursor.fetchone()
        
        return result[0] if result else "Unknown"
    
    def _parse_price_recommendation_response(self, llm_response: str, calculated_price: float) -> Dict:
        """
        Parse LLM price recommendation response
        
        Args:
            llm_response: LLM response text
            calculated_price: Previously calculated optimal price
            
        Returns:
            Parsed response dictionary
        """
        recommendation_type = "ACCEPT"
        recommended_price = calculated_price
        justification = ""
        risks = []
        implementation = ""
        
        # Extract sections
        sections = {
            "ASSESSMENT": "",
            "PRICE_RECOMMENDATION": "",
            "RECOMMENDED_PRICE": "",
            "JUSTIFICATION": "",
            "RISKS": [],
            "IMPLEMENTATION": ""
        }
        
        current_section = None
        
        for line in llm_response.strip().split("\n"):
            line = line.strip()
            
            if not line:
                continue
                
            # Check for section headers
            if line.startswith("ASSESSMENT:"):
                current_section = "ASSESSMENT"
                sections[current_section] = line.replace("ASSESSMENT:", "").strip()
                continue
            elif line.startswith("PRICE_RECOMMENDATION:"):
                current_section = "PRICE_RECOMMENDATION"
                sections[current_section] = line.replace("PRICE_RECOMMENDATION:", "").strip()
                continue
            elif line.startswith("RECOMMENDED_PRICE:"):
                current_section = "RECOMMENDED_PRICE"
                sections[current_section] = line.replace("RECOMMENDED_PRICE:", "").strip()
                continue
            elif line.startswith("JUSTIFICATION:"):
                current_section = "JUSTIFICATION"
                continue
            elif line.startswith("RISKS:"):
                current_section = "RISKS"
                continue
            elif line.startswith("IMPLEMENTATION:"):
                current_section = "IMPLEMENTATION"
                continue
                
            # Add line to current section
            if current_section and current_section in sections:
                if current_section == "RISKS":
                    if line.startswith("- "):
                        sections[current_section].append(line[2:])
                elif current_section in ["JUSTIFICATION", "IMPLEMENTATION"]:
                    if sections[current_section]:
                        sections[current_section] += " " + line
                    else:
                        sections[current_section] = line
        
        # Process recommendation type
        rec_type = sections["PRICE_RECOMMENDATION"].upper()
        if "ADJUST_UP" in rec_type:
            recommendation_type = "ADJUST_UP"
        elif "ADJUST_DOWN" in rec_type:
            recommendation_type = "ADJUST_DOWN"
        else:
            recommendation_type = "ACCEPT"
        
        # Process recommended price
        try:
            price_text = sections["RECOMMENDED_PRICE"]
            # Extract numeric price value
            import re
            price_match = re.search(r'\$?(\d+\.?\d*)', price_text)
            if price_match:
                recommended_price = float(price_match.group(1))
            
            # If no specific price is mentioned but there's a recommendation type
            if recommendation_type == "ADJUST_UP" and recommended_price == calculated_price:
                recommended_price = calculated_price * 1.05  # Default 5% increase
            elif recommendation_type == "ADJUST_DOWN" and recommended_price == calculated_price:
                recommended_price = calculated_price * 0.95  # Default 5% decrease
        except:
            recommended_price = calculated_price
        
        # Process other sections
        justification = sections["JUSTIFICATION"]
        risks = sections["RISKS"]
        implementation = sections["IMPLEMENTATION"]
        
        return {
            "recommendation_type": recommendation_type,
            "recommended_price": recommended_price,
            "justification": justification,
            "risks": risks,
            "implementation": implementation
        }
    
    def _call_ollama_api(self, prompt: str) -> str:
        """
        Call Ollama API to get LLM response
        
        Args:
            prompt: The formatted prompt to send to Ollama
            
        Returns:
            LLM response text
        """
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Low temperature for more deterministic responses
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                self.logger.error(f"API call failed with status code {response.status_code}: {response.text}")
                return ""
        except Exception as e:
            self.logger.error(f"Error calling Ollama API: {e}")
            return ""
    
    def update_price(self, product_id: int, store_id: int, new_price: float, reason: str = None) -> bool:
        """
        Update the price of a product in the database
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            new_price: New price to set
            reason: Reason for price change
            
        Returns:
            Success status
        """
        # Get current price
        pricing_data = self.get_current_pricing_data(product_id, store_id)
        
        if not pricing_data:
            return False
            
        current_price = pricing_data["current_price"]
        
        # Calculate price change percentage
        price_change_pct = (new_price - current_price) / current_price if current_price > 0 else 0
        
        # Update database
        query = """
        UPDATE pricing_optimization
        SET Price = ?
        WHERE Product_ID = ? AND Store_ID = ?
        """
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(query, (new_price, product_id, store_id))
            self.db_conn.commit()
            
            # Publish price change notification
            self._publish_price_change(product_id, store_id, current_price, new_price, price_change_pct, reason)
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating price: {e}")
            return False
    
    def _publish_price_change(self, product_id: int, store_id: int, 
                            old_price: float, new_price: float, 
                            change_pct: float, reason: str = None):
        """
        Publish a price change notification
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            old_price: Previous price
            new_price: New price
            change_pct: Percentage change
            reason: Reason for price change
        """
        if not self.message_bus:
            return
            
        change_type = "increase" if change_pct > 0 else "decrease"
        
        message = {
            "type": "price_change",
            "product_id": product_id,
            "store_id": store_id,
            "old_price": old_price,
            "new_price": new_price,
            "change_pct": change_pct,
            "change_type": change_type,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("pricing_updates", message)
    
    def handle_pricing_data_request(self, params: Dict) -> Dict:
        """
        Handle pricing data request from other agents
        
        Args:
            params: Request parameters
            
        Returns:
            Pricing data
        """
        product_id = params.get("product_id")
        store_id = params.get("store_id")
        
        if not product_id or not store_id:
            return {"error": "Missing product_id or store_id"}
            
        # Get pricing data
        pricing_data = self.get_current_pricing_data(product_id, store_id)
        
        if not pricing_data:
            return {"error": "No pricing data found"}
            
        return pricing_data
    
    def handle_demand_update(self, message: Dict):
        """
        Handle demand update from the Demand Forecasting Agent
        
        Args:
            message: Message containing demand update
        """
        update_type = message.get("type")
        product_id = message.get("product_id")
        store_id = message.get("store_id")
        
        if not product_id or not store_id:
            return
            
        if update_type == "demand_spike":
            # Handle demand spike - consider price increase
            self._handle_demand_spike(product_id, store_id, message)
        elif update_type == "demand_drop":
            # Handle demand drop - consider price decrease
            self._handle_demand_drop(product_id, store_id, message)
    
    def _handle_demand_spike(self, product_id: int, store_id: int, message: Dict):
        """
        Handle demand spike notification
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            message: Demand spike message
        """
        magnitude = message.get("magnitude", 0)
        confidence = message.get("confidence", 0)
        
        # Only respond to significant spikes with reasonable confidence
        if magnitude < 15 or confidence < 60:
            return
            
        # Get inventory status
        inventory_data = self._get_inventory_data(product_id, store_id)
        
        if not inventory_data:
            return
            
        stock_level = inventory_data.get("stock_level", 0)
        reorder_point = inventory_data.get("reorder_point", 0)
        
        # Check if stock level is low relative to reorder point
        if stock_level < reorder_point * 1.5:
            # Stock is relatively low - consider price increase to moderate demand
            pricing_data = self.get_current_pricing_data(product_id, store_id)
            
            if not pricing_data:
                return
                
            current_price = pricing_data["current_price"]
            competitor_price = pricing_data["competitor_price"]
            
            # Only increase price if we're not already higher than competitors
            if current_price < competitor_price:
                # Calculate price recommendation
                optimal_price = self.calculate_optimal_price(product_id, store_id)
                
                if "recommended_price" in optimal_price:
                    # Publish pricing recommendation with demand spike context
                    self._publish_pricing_recommendation(
                        product_id, store_id,
                        optimal_price,
                        "demand_spike",
                        f"Demand spike of {magnitude}% with low inventory"
                    )
    
    def _handle_demand_drop(self, product_id: int, store_id: int, message: Dict):
        """
        Handle demand drop notification
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            message: Demand drop message
        """
        magnitude = message.get("magnitude", 0)
        confidence = message.get("confidence", 0)
        
        # Only respond to significant drops with reasonable confidence
        if magnitude < 15 or confidence < 60:
            return
            
        # Get inventory status
        inventory_data = self._get_inventory_data(product_id, store_id)
        
        if not inventory_data:
            return
            
        stock_level = inventory_data.get("stock_level", 0)
        reorder_point = inventory_data.get("reorder_point", 0)
        
        # Check if stock level is high relative to reorder point
        if stock_level > reorder_point * 2:
            # Stock is relatively high - consider price decrease to stimulate demand
            pricing_data = self.get_current_pricing_data(product_id, store_id)
            
            if not pricing_data:
                return
                
            current_price = pricing_data["current_price"]
            storage_cost = pricing_data["storage_cost"]
            
            # Ensure we maintain minimum profit margin
            min_price = storage_cost * (1 + self.min_margin)
            
            if current_price > min_price * 1.1:  # At least 10% above min price
                # Calculate discount recommendation
                discount_pct = min(0.15, magnitude / 100)  # Cap at 15% discount
                new_price = max(min_price, current_price * (1 - discount_pct))
                
                # Create recommendation message
                recommendation = {
                    "product_id": product_id,
                    "store_id": store_id,
                    "current_price": current_price,
                    "recommended_price": new_price,
                    "price_change": -discount_pct,
                    "recommendation_type": "ADJUST_DOWN",
                    "justification": f"Demand drop of {magnitude}% with high inventory level",
                    "risks": ["Potential margin reduction", "May not fully address demand drop"],
                    "implementation": "Implement immediately to address demand drop",
                    "demand_trend": "Decreasing",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Publish recommendation
                self._publish_pricing_recommendation(
                    product_id, store_id,
                    recommendation,
                    "demand_drop",
                    f"Demand drop of {magnitude}% with high inventory"
                )
    
    def handle_inventory_update(self, message: Dict):
        """
        Handle inventory update from the Inventory Management Agent
        
        Args:
            message: Message containing inventory update
        """
        update_type = message.get("type")
        product_id = message.get("product_id")
        store_id = message.get("store_id")
        
        if not product_id or not store_id:
            return
            
        if update_type == "low_stock":
            # Handle low stock notification - consider price increase
            self._handle_low_stock(product_id, store_id, message)
        elif update_type == "excess_stock":
            # Handle excess stock notification - consider price decrease
            self._handle_excess_stock(product_id, store_id, message)
    
    def _handle_low_stock(self, product_id: int, store_id: int, message: Dict):
        """
        Handle low stock notification
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            message: Low stock message
        """
        # Not implemented in this sample
        pass
    
    def _handle_excess_stock(self, product_id: int, store_id: int, message: Dict):
        """
        Handle excess stock notification
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            message: Excess stock message
        """
        # Not implemented in this sample
        pass
    
    def handle_competitor_update(self, message: Dict):
        """
        Handle competitor update notification
        
        Args:
            message: Message containing competitor update
        """
        update_type = message.get("type")
        product_id = message.get("product_id")
        competitor_price = message.get("competitor_price")
        
        if not product_id or not competitor_price:
            return
            
        # Get all stores that carry this product
        stores = self._get_stores_with_product(product_id)
        
        for store_id in stores:
            # Get current pricing data
            pricing_data = self.get_current_pricing_data(product_id, store_id)
            
            if not pricing_data:
                continue
                
            current_price = pricing_data["current_price"]
            
            # Calculate price difference as percentage
            price_diff_pct = (current_price - competitor_price) / competitor_price
            
            # Respond if our price is significantly different from competitor
            if abs(price_diff_pct) > self.competitor_threshold:
                self._respond_to_competitor_price(product_id, store_id, 
                                                current_price, competitor_price, price_diff_pct)
    
    def _get_stores_with_product(self, product_id: int) -> List[int]:
        """
        Get all stores that carry a specific product
        
        Args:
            product_id: The product identifier
            
        Returns:
            List of store IDs
        """
        query = """
        SELECT DISTINCT Store_ID 
        FROM pricing_optimization
        WHERE Product_ID = ?
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (product_id,))
        results = cursor.fetchall()
        
        return [row[0] for row in results]
    
    def _respond_to_competitor_price(self, product_id: int, store_id: int, 
                                   current_price: float, competitor_price: float, 
                                   price_diff_pct: float):
        """
        Respond to significant competitor price difference
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            current_price: Current product price
            competitor_price: Competitor price
            price_diff_pct: Percentage difference
        """
        # Get pricing data and inventory context
        pricing_data = self.get_current_pricing_data(product_id, store_id)
        inventory_data = self._get_inventory_data(product_id, store_id)
        
        if not pricing_data:
            return
            
        # Create context for LLM reasoning
        elasticity = pricing_data['elasticity']
        storage_cost = pricing_data['storage_cost']
        
        # Get inventory context
        stock_level = inventory_data.get('stock_level', 'unknown') if inventory_data else 'unknown'
        reorder_point = inventory_data.get('reorder_point', 'unknown') if inventory_data else 'unknown'
        
        # Create prompt for LLM
        prompt = f"""
        You are a pricing optimization expert for a retail chain. I need your recommendation on responding to a competitor price change:
        
        Product ID: {product_id}
        Store ID: {store_id}
        
        Current situation:
        - Our current price: ${current_price:.2f}
        - Competitor's price: ${competitor_price:.2f}
        - Price difference: {price_diff_pct*100:.1f}% {'higher' if price_diff_pct > 0 else 'lower'} than competitor
        - Storage/cost per unit: ${storage_cost:.2f}
        - Price elasticity: {elasticity:.2f}
        - Current stock level: {stock_level} units
        - Reorder point: {reorder_point} units
        
        How should we respond to this competitor pricing? Please consider:
        1. Whether we should match the competitor's price, partially adjust our price, or maintain our current price
        2. The impact on our sales volume and profitability
        3. Our current inventory position
        
        Format your response as follows:
        RECOMMENDATION: [one of: MATCH_COMPETITOR, PARTIAL_ADJUSTMENT, MAINTAIN_PRICE]
        RECOMMENDED_PRICE: [your price recommendation]
        
        JUSTIFICATION:
        [detailed justification]
        
        IMPLEMENTATION:
        [implementation recommendation]
        """
        
        # Call LLM
        llm_response = self._call_ollama_api(prompt)
        
        # Parse LLM response
        recommendation = self._parse_competitor_response(llm_response, current_price, competitor_price)
        
        # Publish pricing recommendation
        self._publish_pricing_recommendation(
            product_id, store_id,
            recommendation,
            "competitor_price",
            f"Competitor price is {price_diff_pct*100:.1f}% {'lower' if price_diff_pct > 0 else 'higher'}"
        )
    
    def _parse_competitor_response(self, llm_response: str, current_price: float, competitor_price: float) -> Dict:
        """
        Parse LLM response for competitor price reaction
        
        Args:
            llm_response: LLM response text
            current_price: Current product price
            competitor_price: Competitor price
            
        Returns:
            Parsed recommendation dictionary
        """
        recommendation_type = "MAINTAIN_PRICE"
        recommended_price = current_price
        justification = ""
        implementation = ""
        
        # Extract sections
        sections = {
            "RECOMMENDATION": "",
            "RECOMMENDED_PRICE": "",
            "JUSTIFICATION": "",
            "IMPLEMENTATION": ""
        }
        
        current_section = None
        
        for line in llm_response.strip().split("\n"):
            line = line.strip()
            
            if not line:
                continue
                
            # Check for section headers
            if line.startswith("RECOMMENDATION:"):
                current_section = "RECOMMENDATION"
                sections[current_section] = line.replace("RECOMMENDATION:", "").strip()
                continue
            elif line.startswith("RECOMMENDED_PRICE:"):
                current_section = "RECOMMENDED_PRICE"
                sections[current_section] = line.replace("RECOMMENDED_PRICE:", "").strip()
                continue
            elif line.startswith("JUSTIFICATION:"):
                current_section = "JUSTIFICATION"
                continue
            elif line.startswith("IMPLEMENTATION:"):
                current_section = "IMPLEMENTATION"
                continue
                
            # Add line to current section
            if current_section and current_section in sections:
                if current_section in ["JUSTIFICATION", "IMPLEMENTATION"]:
                    if sections[current_section]:
                        sections[current_section] += " " + line
                    else:
                        sections[current_section] = line
        
        # Process recommendation type
        rec_type = sections["RECOMMENDATION"].upper()
        if "MATCH_COMPETITOR" in rec_type:
            recommendation_type = "MATCH_COMPETITOR"
            recommended_price = competitor_price
        elif "PARTIAL_ADJUSTMENT" in rec_type:
            recommendation_type = "PARTIAL_ADJUSTMENT"
            # Calculate a price halfway between current and competitor
            recommended_price = (current_price + competitor_price) / 2
        else:
            recommendation_type = "MAINTAIN_PRICE"
            recommended_price = current_price
        
        # Process recommended price if specified
        try:
            price_text = sections["RECOMMENDED_PRICE"]
            # Extract numeric price value
            import re
            price_match = re.search(r'\$?(\d+\.?\d*)', price_text)
            if price_match:
                recommended_price = float(price_match.group(1))
        except:
            # Use the default based on recommendation type
            pass
        
        # Process other sections
        justification = sections["JUSTIFICATION"]
        implementation = sections["IMPLEMENTATION"]
        
        # Convert recommendation type to standard format
        if recommendation_type == "MATCH_COMPETITOR":
            std_type = "ADJUST_DOWN" if current_price > competitor_price else "ADJUST_UP"
        elif recommendation_type == "PARTIAL_ADJUSTMENT":
            std_type = "ADJUST_DOWN" if current_price > recommended_price else "ADJUST_UP"
        else:
            std_type = "MAINTAIN"
        
        # Create recommendation dictionary
        parsed_recommendation = {
            "product_id": "TBD",  # Will be set by caller
            "store_id": "TBD",    # Will be set by caller
            "current_price": current_price,
            "competitor_price": competitor_price,
            "recommended_price": recommended_price,
            "price_change": (recommended_price - current_price) / current_price,
            "recommendation_type": std_type,
            "justification": justification,
            "risks": ["Potential customer confusion", "May impact brand perception"],
            "implementation": implementation,
            "timestamp": datetime.now().isoformat()
        }
        
        return parsed_recommendation
    
    def _publish_pricing_recommendation(self, product_id: int, store_id: int, 
                                      recommendation: Dict, source: str, reason: str):
        """
        Publish pricing recommendation to message bus
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            recommendation: Recommendation data
            source: Source of recommendation
            reason: Reason for recommendation
        """
        if not self.message_bus:
            return
            
        # Ensure product and store IDs are set
        recommendation["product_id"] = product_id
        recommendation["store_id"] = store_id
        recommendation["source"] = source
        recommendation["reason"] = reason
        
        # Add messaging fields
        message = {
            "type": "pricing_recommendation",
            **recommendation
        }
        
        self.message_bus.publish("pricing_recommendations", message)
    
    def run_periodic_pricing_optimization(self):
        """
        Run periodic pricing optimization for all products
        """
        # This would typically run for all products, but for demonstration
        # we'll just get a sample of products
        sample_products = self._get_sample_products(limit=10)
        
        recommendations = []
        
        for product_id, store_id in sample_products:
            # Skip products that have had recent price changes
            if self._has_recent_price_change(product_id, store_id):
                continue
                
            # Calculate optimal price
            optimization_result = self.calculate_optimal_price(product_id, store_id)
            
            # Check if recommendation suggests price change
            if "recommended_price" in optimization_result:
                recommended_price = optimization_result["recommended_price"]
                current_price = optimization_result["current_price"]
                
                # Only consider significant price changes
                price_change_pct = abs((recommended_price - current_price) / current_price)
                
                if price_change_pct > 0.02:  # At least 2% change
                    # Add to recommendations
                    recommendations.append(optimization_result)
                    
                    # Publish recommendation
                    self._publish_pricing_recommendation(
                        product_id, store_id,
                        optimization_result,
                        "periodic_optimization",
                        "Regular pricing optimization"
                    )
        
        return {
            "recommendations": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_sample_products(self, limit: int = 10) -> List[Tuple[int, int]]:
        """
        Get a sample of products for processing
        
        Args:
            limit: Maximum number of products to retrieve
            
        Returns:
            List of (product_id, store_id) tuples
        """
        query = """
        SELECT DISTINCT Product_ID, Store_ID 
        FROM pricing_optimization
        LIMIT ?
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        
        return results
    
    def _has_recent_price_change(self, product_id: int, store_id: int, days: int = 7) -> bool:
        """
        Check if a product has had a recent price change
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            days: Number of days to consider recent
            
        Returns:
            True if there has been a recent price change
        """
        # This would typically check a price history table
        # For this sample, we'll assume no recent changes
        return False