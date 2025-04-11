# inventory_agent.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import logging
import re
from typing import Dict, List, Tuple, Any, Optional

class InventoryAgent:
    """
    Agent responsible for inventory management, reordering,
    and stock optimization across stores.
    """
    
    def __init__(self, config: Dict, db_path: str, ollama_base_url: str):
        """
        Initialize the Inventory Management Agent
        
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
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def connect_to_message_bus(self, message_bus):
        """Connect to the inter-agent communication bus"""
        self.message_bus = message_bus
        # Subscribe to relevant topics
        self.message_bus.subscribe("demand_forecast_updated", self.handle_demand_update)
        self.message_bus.subscribe("price_change", self.handle_price_change)
        self.message_bus.subscribe("supplier_update", self.handle_supplier_update)
    
    def handle_demand_update(self, message: Dict):
        """
        Handle updates from the Demand Forecasting Agent
        
        Args:
            message: Message containing demand update information
        """
        try:
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            
            if not product_id or not store_id:
                self.logger.warning("Incomplete demand update message")
                return
                
            # Check if reorder is needed based on new demand forecast
            reorder_needed, context = self.check_reorder_needed(product_id, store_id)
            
            if reorder_needed:
                # Calculate order quantity
                order_quantity = self.calculate_optimal_order_quantity(product_id, store_id)
                
                # Publish reorder recommendation
                if order_quantity > 0:
                    self.publish_reorder_recommendation(product_id, store_id, order_quantity, context)
        except Exception as e:
            self.logger.error(f"Error in handle_demand_update: {e}")
    
    def handle_price_change(self, message: Dict):
        """
        Handle updates from the Pricing Optimization Agent
        
        Args:
            message: Message containing price change information
        """
        try:
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            price_change = message.get("price_change", 0)
            
            if not product_id or not store_id:
                self.logger.warning("Incomplete price change message")
                return
                
            # If significant price decrease, check if we need more inventory
            if price_change < -0.05:  # 5% price decrease
                self.handle_demand_update({
                    "product_id": product_id,
                    "store_id": store_id,
                    "type": "potential_demand_increase"
                })
        except Exception as e:
            self.logger.error(f"Error in handle_price_change: {e}")
    
    def handle_supplier_update(self, message: Dict):
        """
        Handle updates from the Supply Chain Agent
        
        Args:
            message: Message containing supplier update information
        """
        try:
            product_id = message.get("product_id")
            lead_time_change = message.get("lead_time_change", 0)
            
            if not product_id or lead_time_change == 0:
                self.logger.warning("Incomplete supplier update message")
                return
                
            # Update database with new lead time information
            if lead_time_change > 0:
                # Lead time increased, check if we need to adjust reorder points
                stores = self.get_stores_with_product(product_id)
                for store_id in stores:
                    # Recalculate reorder point
                    self.update_reorder_point(product_id, store_id, lead_time_change)
        except Exception as e:
            self.logger.error(f"Error in handle_supplier_update: {e}")

    def get_current_inventory(self, product_id: int, store_id: int) -> Dict:
        """
        Get current inventory data for a specific product at a store
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Dictionary with inventory details
        """
        query = """
        SELECT 
            Stock_Levels, 
            Supplier_Lead_Time_days, 
            Stockout_Frequency, 
            Reorder_Point, 
            Expiry_Date, 
            Warehouse_Capacity, 
            Order_Fulfillment_Time_days
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
            
        inventory_data = {
            "product_id": product_id,
            "store_id": store_id,
            "stock_level": result[0],
            "supplier_lead_time": result[1],
            "stockout_frequency": result[2],
            "reorder_point": result[3],
            "expiry_date": result[4],
            "warehouse_capacity": result[5],
            "fulfillment_time": result[6],
            "timestamp": datetime.now().isoformat()
        }
        
        return inventory_data

    def get_sales_velocity(self, product_id: int, store_id: int, days: int = 30) -> float:
        """
        Calculate the sales velocity (average daily sales) for a product
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            days: Number of days to look back
            
        Returns:
            Average daily sales quantity
        """
        today = datetime.now()
        start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        
        query = """
        SELECT 
            AVG(Sales_Quantity) as avg_daily_sales
        FROM 
            demand_forecasting
        WHERE 
            Product_ID = ? AND 
            Store_ID = ? AND
            Date >= ?
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (product_id, store_id, start_date))
        result = cursor.fetchone()
        
        return result[0] if result and result[0] is not None else 0.0

    def calculate_days_of_supply(self, product_id: int, store_id: int) -> float:
        """
        Calculate how many days the current inventory will last
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Days of supply remaining
        """
        inventory = self.get_current_inventory(product_id, store_id)
        if not inventory or inventory["stock_level"] == 0:
            return 0
            
        sales_velocity = self.get_sales_velocity(product_id, store_id)
        if sales_velocity == 0:
            return float('inf')  # No sales, infinite days of supply
            
        return inventory["stock_level"] / sales_velocity

    def check_reorder_needed(self, product_id: int, store_id: int) -> Tuple[bool, Dict]:
        """
        Determine if a product needs to be reordered
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Tuple of (reorder_needed, context_data)
        """
        inventory = self.get_current_inventory(product_id, store_id)
        if not inventory:
            return False, {}
            
        days_of_supply = self.calculate_days_of_supply(product_id, store_id)
        lead_time = inventory["supplier_lead_time"]
        
        # Get future demand forecast from Demand Forecasting Agent
        future_demand = self.request_forecast(product_id, store_id)
        
        # Check if stock will cover lead time plus safety buffer
        safety_days = self.config.get("safety_buffer_days", 5)
        critical_threshold = lead_time + safety_days
        
        reorder_needed = days_of_supply <= critical_threshold
        
        context = {
            "product_id": product_id,
            "store_id": store_id,
            "current_stock": inventory["stock_level"],
            "days_of_supply": days_of_supply,
            "lead_time": lead_time,
            "safety_buffer": safety_days,
            "reorder_point": inventory["reorder_point"],
            "future_demand": future_demand,
            "stockout_history": inventory["stockout_frequency"],
            "warehouse_capacity": inventory["warehouse_capacity"]
        }
        
        return reorder_needed, context

    def request_forecast(self, product_id: int, store_id: int) -> Dict:
        """
        Request demand forecast from the Demand Forecasting Agent
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Forecast data with safe default values
        """
        try:
            if self.message_bus:
                # Attempt to get forecast from message bus
                response = self.message_bus.request(
                    "demand_agent", 
                    "get_forecast", 
                    {"product_id": product_id, "store_id": store_id}
                )
                
                # Ensure response is a dictionary with safe defaults
                if response is None:
                    self.logger.warning(f"No forecast received for product {product_id}, store {store_id}")
                    return {"annual": 0, "confidence": 0}
                
                # Validate key fields
                return {
                    "annual": response.get("annual", 0),
                    "confidence": response.get("confidence", 0)
                }
            else:
                # Fallback if message bus not available
                self.logger.info("Message bus not available, using default forecast")
                return {"annual": 0, "confidence": 0}
        except Exception as e:
            # Log the error and return safe defaults
            self.logger.error(f"Error requesting forecast: {e}")
            return {"annual": 0, "confidence": 0}

    def calculate_optimal_order_quantity(self, product_id: int, store_id: int) -> int:
        """
        Calculate the optimal order quantity using Economic Order Quantity model
        with adjustments based on LLM reasoning
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Recommended order quantity
        """
        # Get base data
        reorder_needed, context = self.check_reorder_needed(product_id, store_id)
        
        if not reorder_needed:
            return 0
        
        # Safely handle potential None or missing context
        if not context:
            self.logger.warning(f"No context available for product {product_id}, store {store_id}")
            return 0
        
        # Get pricing data with safe fallback
        pricing_data = self.request_pricing_data(product_id, store_id) or {}
        
        # Safely calculate annual demand
        sales_velocity = self.get_sales_velocity(product_id, store_id)
        future_demand = context.get("future_demand", {})
        annual_demand = future_demand.get("annual", 365 * sales_velocity)
        
        # Safely retrieve configuration and pricing parameters
        order_cost = self.config.get("default_order_cost", 20)  # Default cost per order
        storage_cost = pricing_data.get("storage_cost", 0.1)  # Default storage cost
        holding_cost = storage_cost * 0.2  # 20% of unit cost as holding cost
        
        # Calculate base order quantity
        try:
            if holding_cost > 0:
                # Economic Order Quantity (EOQ) formula
                eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
                base_quantity = max(1, int(np.round(eoq)))
            else:
                # Fallback calculation based on lead time and sales velocity
                base_quantity = max(1, int(context.get("lead_time", 1) * sales_velocity * 1.5))
        except Exception as e:
            self.logger.error(f"Error calculating base order quantity: {e}")
            base_quantity = max(1, int(context.get("lead_time", 1) * sales_velocity * 1.5))
        
        # Safely get context parameters
        warehouse_capacity = context.get("warehouse_capacity", float('inf'))
        current_stock = context.get("current_stock", 0)
        
        # Use LLM to refine the order quantity recommendation
        try:
            refined_quantity = self.refine_with_llm(base_quantity, context, pricing_data)
        except Exception as e:
            self.logger.error(f"Error refining order quantity with LLM: {e}")
            refined_quantity = base_quantity
        
        # Apply final constraints
        min_order = max(1, int(context.get("lead_time", 1) * sales_velocity))
        max_order = min(
            int(warehouse_capacity - current_stock),
            int(min_order * 5)  # Cap at 5x the minimum reasonable order
        )
        
        # Final quantity validation
        final_quantity = max(min_order, min(refined_quantity, max_order))
        
        return final_quantity

    def request_pricing_data(self, product_id: int, store_id: int) -> Dict:
        """
        Request pricing data from the Pricing Optimization Agent
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            
        Returns:
            Pricing data
        """
        if self.message_bus:
            try:
                response = self.message_bus.request(
                    "pricing_agent", 
                    "get_pricing_data", 
                    {"product_id": product_id, "store_id": store_id}
                )
                return response if response is not None else {}
            except Exception as e:
                self.logger.error(f"Error requesting pricing data: {e}")
                return {}
        else:
            # Fallback if message bus not available
            query = """
            SELECT 
                Price, 
                Competitor_Prices,
                Discounts, 
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
                return {}
                
            return {
                "price": result[0],
                "competitor_price": result[1],
                "discount": result[2],
                "storage_cost": result[3],
                "elasticity": result[4]
            }

    def refine_with_llm(self, base_quantity: int, inventory_context: Dict, pricing_data: Dict) -> int:
        """
        Use LLM to refine order quantity recommendation based on context
        
        Args:
            base_quantity: Base order quantity from EOQ formula
            inventory_context: Inventory context data
            pricing_data: Pricing data
            
        Returns:
            Refined order quantity
        """
        try:
            prompt = self._create_order_quantity_prompt(base_quantity, inventory_context, pricing_data)
            
            response = self._call_ollama_api(prompt)
            
            # Extract the recommended quantity from the LLM response
            try:
                # Try to find a simple number in the response
                quantity_matches = re.findall(r"recommended quantity[:\s]+(\d+)", response.lower())
                if quantity_matches:
                    refined_quantity = int(quantity_matches[0])
                else:
                    # Fallback to searching for any number
                    quantity_matches = re.findall(r"\b(\d+)\b", response)
                    if quantity_matches:
                        refined_quantity = int(quantity_matches[0])
                    else:
                        refined_quantity = base_quantity
            except Exception as e:
                self.logger.error(f"Error parsing LLM response: {e}")
                refined_quantity = base_quantity
                
            # Apply reasonability constraints
            min_order = max(1, int(inventory_context["lead_time"] * self.get_sales_velocity(
                inventory_context["product_id"], inventory_context["store_id"])))
            max_order = min(
                inventory_context["warehouse_capacity"] - inventory_context["current_stock"],
                int(min_order * 5)  # Cap at 5x the minimum reasonable order
            )
            
            return max(min_order, min(refined_quantity, max_order))
            
        except Exception as e:
            self.logger.error(f"Error refining order quantity with LLM: {e}")
            return base_quantity

    def _create_order_quantity_prompt(self, base_quantity: int, inventory_context: Dict, pricing_data: Dict) -> str:
        """
        Create prompt for LLM to refine order quantity
        
        Args:
            base_quantity: Base order quantity from EOQ formula
            inventory_context: Inventory context data
            pricing_data: Pricing data
            
        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""
        You are an inventory optimization expert. Based on the following information,
        recommend an optimal order quantity for the product.
        
        Product Information:
        - Product ID: {inventory_context['product_id']}
        - Store ID: {inventory_context['store_id']}
        - Current stock level: {inventory_context['current_stock']} units
        - Days of supply remaining: {inventory_context['days_of_supply']:.1f} days
        - Supplier lead time: {inventory_context['lead_time']} days
        - Historical stockout frequency: {inventory_context['stockout_history']} times
        - Warehouse remaining capacity: {inventory_context['warehouse_capacity'] - inventory_context['current_stock']} units
        
        Pricing Information:
        - Current price: ${pricing_data.get('price', 'N/A')}
        - Competitor price: ${pricing_data.get('competitor_price', 'N/A')}
        - Storage cost per unit: ${pricing_data.get('storage_cost', 'N/A')}
        - Price elasticity index: {pricing_data.get('elasticity', 'N/A')}
        
        The base Economic Order Quantity (EOQ) calculation suggests ordering {base_quantity} units.
        
        Additional Context:
        - Safety buffer policy: {inventory_context['safety_buffer']} days
        - Standard reorder point: {inventory_context['reorder_point']} units
        
        Please analyze this information and recommend a final order quantity,
        explaining your reasoning. Your response should include:
        1. Recommended quantity: [your recommendation]
        2. Key factors that influenced your decision
        3. Any risks associated with this order quantity
        """
        
        return prompt

    def _call_ollama_api(self, prompt: str) -> str:
        """
        Call Ollama API to get LLM response
        
        Args:
            prompt: The formatted prompt to send to Ollama
            
        Returns:
            LLM response text
        """
        try:
            url = f"{self.ollama_url}/api/generate"
            
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for more deterministic responses
                    "top_p": 0.9
                }
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                self.logger.error(f"Ollama API call failed: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            self.logger.error(f"Error calling Ollama API: {e}")
            return ""

    def get_stores_with_product(self, product_id: int) -> List[int]:
        """
        Get all stores that carry a specific product
        
        Args:
            product_id: The product identifier
            
        Returns:
            List of store IDs
        """
        query = """
        SELECT DISTINCT Store_ID 
        FROM inventory_monitoring
        WHERE Product_ID = ?
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (product_id,))
        results = cursor.fetchall()
        
        return [row[0] for row in results]

    def update_reorder_point(self, product_id: int, store_id: int, lead_time_change: int):
        """
        Update reorder point based on lead time change
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            lead_time_change: The change in lead time in days
        """
        # Get current inventory data
        inventory = self.get_current_inventory(product_id, store_id)
        if not inventory:
            self.logger.warning(f"No inventory data found for product {product_id}, store {store_id}")
            return
            
        # Calculate new reorder point
        sales_velocity = self.get_sales_velocity(product_id, store_id)
        safety_stock = self.config.get("safety_stock_factor", 1.5) * sales_velocity * lead_time_change
        
        new_reorder_point = inventory["reorder_point"] + int(safety_stock)
        
        # Update database
        query = """
        UPDATE inventory_monitoring
        SET Reorder_Point = ?
        WHERE Product_ID = ? AND Store_ID = ?
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (new_reorder_point, product_id, store_id))
        self.db_conn.commit()
        
        # Publish update
        self.publish_reorder_point_update(product_id, store_id, new_reorder_point)

    def publish_reorder_recommendation(self, product_id: int, store_id: int, quantity: int, context: Dict):
        """
        Publish a reorder recommendation to the message bus
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            quantity: Recommended order quantity
            context: Additional context for the recommendation
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "reorder_recommendation",
            "product_id": product_id,
            "store_id": store_id,
            "quantity": quantity,
            "days_of_supply": context.get("days_of_supply", 0),
            "lead_time": context.get("lead_time", 0),
            "current_stock": context.get("current_stock", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("inventory_recommendations", message)

    def publish_reorder_point_update(self, product_id: int, store_id: int, new_reorder_point: int):
        """
        Publish a reorder point update to the message bus
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            new_reorder_point: Updated reorder point
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "reorder_point_update",
            "product_id": product_id,
            "store_id": store_id,
            "reorder_point": new_reorder_point,
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("inventory_updates", message)

    def run_periodic_inventory_check(self):
        """
        Perform a periodic check of all inventory items
        """
        # Get all product-store combinations
        query = """
        SELECT DISTINCT Product_ID, Store_ID 
        FROM inventory_monitoring
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        recommendations = []
        
        for product_id, store_id in results:
            try:
                # Check if reorder is needed
                reorder_needed, context = self.check_reorder_needed(product_id, store_id)
                
                if reorder_needed:
                    # Calculate order quantity
                    order_quantity = self.calculate_optimal_order_quantity(product_id, store_id)
                    
                    if order_quantity > 0:
                        recommendations.append({
                            "product_id": product_id,
                            "store_id": store_id,
                            "quantity": order_quantity,
                            "days_of_supply": context.get("days_of_supply", 0),
                            "lead_time": context.get("lead_time", 0)
                        })
                    
                        # Publish recommendation
                        self.publish_reorder_recommendation(product_id, store_id, order_quantity, context)
            except Exception as e:
                self.logger.error(f"Error processing inventory check for product {product_id}, store {store_id}: {e}")
        
        return recommendations

    def cleanup(self):
        """
        Perform cleanup operations when the agent is shutting down
        """
        try:
            if self.db_conn:
                self.db_conn.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")