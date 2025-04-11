# supply_chain_agent.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import defaultdict

class SupplyChainAgent:
    """
    Agent responsible for coordinating suppliers, managing orders,
    and optimizing the supply chain.
    """
    
    def __init__(self, config: Dict, db_path: str, ollama_base_url: str):
        """
        Initialize the Supply Chain Agent
        
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
        self.logger = logging.getLogger("supply_chain_agent")
        
        # Track pending orders
        self.pending_orders = {}
        
        # Configure supply chain parameters
        self.lead_time_buffer = config.get("lead_time_buffer", 3)  # Extra buffer days for supplier delays
        self.consolidation_threshold = config.get("consolidation_threshold", 3)  # Min products to consolidate orders
        self.expedite_cost_multiplier = config.get("expedite_cost_multiplier", 1.5)  # Cost multiplier for expedited orders
        
    def connect_to_message_bus(self, message_bus):
        """Connect to the inter-agent communication bus"""
        self.message_bus = message_bus
        
        # Subscribe to relevant topics
        self.message_bus.subscribe("inventory_commands", self.handle_inventory_command)
        self.message_bus.subscribe("approved_recommendations", self.handle_approved_recommendation)
        self.message_bus.subscribe("coordinated_actions", self.handle_coordinated_action)
        
    def get_supplier_info(self, supplier_id: int) -> Dict:
        """
        Get information about a supplier
        
        Args:
            supplier_id: The supplier identifier
            
        Returns:
            Dictionary with supplier details
        """
        # In a real implementation, this would query a supplier database
        # For this demo, we'll simulate supplier data
        suppliers = {
            1: {
                "name": "Main Electronics Supplier",
                "reliability": 0.95,
                "avg_lead_time": 7,
                "min_order_value": 1000,
                "product_categories": ["electronics"]
            },
            2: {
                "name": "Fashion Wholesale Inc",
                "reliability": 0.92,
                "avg_lead_time": 10,
                "min_order_value": 2000,
                "product_categories": ["clothing"]
            },
            3: {
                "name": "Grocery Supply Co",
                "reliability": 0.98,
                "avg_lead_time": 3,
                "min_order_value": 500,
                "product_categories": ["groceries"]
            },
            4: {
                "name": "Home Goods Distribution",
                "reliability": 0.94,
                "avg_lead_time": 8,
                "min_order_value": 1500,
                "product_categories": ["home"]
            },
            5: {
                "name": "General Merchandise Supply",
                "reliability": 0.90,
                "avg_lead_time": 9,
                "min_order_value": 1000,
                "product_categories": ["other"]
            }
        }
        
        return suppliers.get(supplier_id, {})
    
    def get_product_supplier(self, product_id: int) -> int:
        """
        Get the supplier for a product
        
        Args:
            product_id: The product identifier
            
        Returns:
            Supplier ID
        """
        # In a real implementation, this would query a product-supplier mapping
        # For this demo, we'll determine supplier based on product ID ranges
        if product_id < 2000:
            return 1  # Electronics supplier
        elif product_id < 4000:
            return 2  # Clothing supplier
        elif product_id < 6000:
            return 3  # Grocery supplier
        elif product_id < 8000:
            return 4  # Home goods supplier
        else:
            return 5  # General merchandise supplier
    
    def get_product_info(self, product_id: int) -> Dict:
        """
        Get information about a product
        
        Args:
            product_id: The product identifier
            
        Returns:
            Dictionary with product details
        """
        # In a real implementation, this would query a product database
        # For this demo, we'll simulate based on product ID
        
        # Get product category
        if product_id < 2000:
            category = "electronics"
        elif product_id < 4000:
            category = "clothing"
        elif product_id < 6000:
            category = "groceries"
        elif product_id < 8000:
            category = "home"
        else:
            category = "other"
            
        # Generate simulated product data
        unit_cost = np.random.uniform(5, 100)
        if category == "electronics":
            unit_cost *= 2  # Electronics are more expensive
        elif category == "groceries":
            unit_cost /= 2  # Groceries are less expensive
            
        # Get supplier
        supplier_id = self.get_product_supplier(product_id)
            
        return {
            "product_id": product_id,
            "category": category,
            "unit_cost": unit_cost,
            "supplier_id": supplier_id,
            "pack_size": np.random.choice([1, 6, 12, 24, 48]),
            "case_size": np.random.choice([1, 4, 6, 12]),
            "min_order_quantity": np.random.choice([1, 5, 10, 20, 50]),
            "is_perishable": category == "groceries" and np.random.random() < 0.7
        }
    
    def get_order_info(self, order_id: str) -> Dict:
        """
        Get information about an order
        
        Args:
            order_id: The order identifier
            
        Returns:
            Dictionary with order details
        """
        # In a real implementation, this would query an orders database
        # For this demo, we'll check our pending orders
        return self.pending_orders.get(order_id, {})
    
    def create_order(self, product_id: int, store_id: int, quantity: int, 
                    is_expedited: bool = False) -> str:
        """
        Create a new order for a product
        
        Args:
            product_id: The product identifier
            store_id: The store identifier
            quantity: Order quantity
            is_expedited: Whether this is an expedited order
            
        Returns:
            Order ID
        """
        # Get supplier information
        supplier_id = self.get_product_supplier(product_id)
        supplier_info = self.get_supplier_info(supplier_id)
        
        # Get product information
        product_info = self.get_product_info(product_id)
        
        # Calculate order details
        unit_cost = product_info.get("unit_cost", 0)
        min_order_quantity = product_info.get("min_order_quantity", 1)
        
        # Adjust quantity to meet minimum order quantity
        adjusted_quantity = max(quantity, min_order_quantity)
        
        # Calculate order value
        order_value = adjusted_quantity * unit_cost
        
        # Apply expedited multiplier if needed
        if is_expedited:
            order_value *= self.expedite_cost_multiplier
            
        # Calculate expected lead time
        avg_lead_time = supplier_info.get("avg_lead_time", 7)
        reliability = supplier_info.get("reliability", 0.9)
        
        # Add buffer based on reliability
        buffer_days = round(self.lead_time_buffer * (1 - reliability) * 10)
        expected_lead_time = avg_lead_time + buffer_days
        
        # Reduce lead time for expedited orders
        if is_expedited:
            expected_lead_time = max(1, round(expected_lead_time * 0.7))
            
        # Generate order ID
        order_id = f"ORD-{product_id}-{store_id}-{int(datetime.now().timestamp())}"
        
        # Create order record
        order = {
            "order_id": order_id,
            "product_id": product_id,
            "store_id": store_id,
            "supplier_id": supplier_id,
            "quantity": adjusted_quantity,
            "unit_cost": unit_cost,
            "total_cost": order_value,
            "is_expedited": is_expedited,
            "order_date": datetime.now().isoformat(),
            "expected_delivery_date": (datetime.now() + timedelta(days=expected_lead_time)).isoformat(),
            "expected_lead_time": expected_lead_time,
            "status": "created",
            "last_updated": datetime.now().isoformat()
        }
        
        # Store in pending orders
        self.pending_orders[order_id] = order
        
        # In a real implementation, this would be saved to a database
        
        # Publish order creation notification
        self._publish_order_creation(order)
        
        return order_id
    
    def _publish_order_creation(self, order: Dict):
        """
        Publish order creation notification
        
        Args:
            order: Order details
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "order_created",
            "order_id": order["order_id"],
            "product_id": order["product_id"],
            "store_id": order["store_id"],
            "quantity": order["quantity"],
            "expected_delivery_date": order["expected_delivery_date"],
            "is_expedited": order["is_expedited"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("supply_chain_updates", message)
    
    def update_order_status(self, order_id: str, new_status: str, 
                          additional_info: Dict = None) -> bool:
        """
        Update the status of an order
        
        Args:
            order_id: The order identifier
            new_status: New order status
            additional_info: Additional information to update
            
        Returns:
            Success status
        """
        if order_id not in self.pending_orders:
            return False
            
        # Update status
        self.pending_orders[order_id]["status"] = new_status
        self.pending_orders[order_id]["last_updated"] = datetime.now().isoformat()
        
        # Update additional fields if provided
        if additional_info:
            for key, value in additional_info.items():
                if key not in ["order_id", "status", "last_updated"]:
                    self.pending_orders[order_id][key] = value
        
        # Publish status update
        self._publish_order_update(self.pending_orders[order_id])
        
        return True
    
    def _publish_order_update(self, order: Dict):
        """
        Publish order update notification
        
        Args:
            order: Updated order details
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "order_updated",
            "order_id": order["order_id"],
            "product_id": order["product_id"],
            "store_id": order["store_id"],
            "status": order["status"],
            "expected_delivery_date": order.get("expected_delivery_date", "unknown"),
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("supply_chain_updates", message)
    
    def check_supplier_performance(self, supplier_id: int) -> Dict:
        """
        Check supplier performance metrics
        
        Args:
            supplier_id: The supplier identifier
            
        Returns:
            Performance metrics
        """
        # In a real implementation, this would analyze order history
        # For this demo, we'll simulate based on supplier info
        supplier_info = self.get_supplier_info(supplier_id)
        
        reliability = supplier_info.get("reliability", 0.9)
        
        # Generate simulated performance metrics
        on_time_delivery_rate = max(0.7, min(1.0, np.random.normal(reliability, 0.05)))
        order_accuracy_rate = max(0.8, min(1.0, np.random.normal(reliability + 0.05, 0.03)))
        quality_rate = max(0.9, min(1.0, np.random.normal(reliability + 0.08, 0.02)))
        
        # Calculate overall score
        overall_score = (on_time_delivery_rate * 0.4 + 
                        order_accuracy_rate * 0.3 + 
                        quality_rate * 0.3)
        
        return {
            "supplier_id": supplier_id,
            "supplier_name": supplier_info.get("name", f"Supplier {supplier_id}"),
            "on_time_delivery_rate": on_time_delivery_rate,
            "order_accuracy_rate": order_accuracy_rate,
            "quality_rate": quality_rate,
            "overall_score": overall_score,
            "timestamp": datetime.now().isoformat()
        }
    
    def consolidate_orders(self, supplier_id: int, store_id: int = None) -> List[Dict]:
        """
        Consolidate multiple orders to the same supplier
        
        Args:
            supplier_id: The supplier identifier
            store_id: Optional store ID to filter by
            
        Returns:
            List of consolidated orders
        """
        # Get all pending orders for this supplier
        supplier_orders = []
        
        for order_id, order in self.pending_orders.items():
            if (order["supplier_id"] == supplier_id and 
                order["status"] == "created" and
                (store_id is None or order["store_id"] == store_id)):
                supplier_orders.append(order)
        
        # Check if we have enough orders to consolidate
        if len(supplier_orders) < self.consolidation_threshold:
            return []
            
        # Group orders by store
        store_orders = defaultdict(list)
        
        for order in supplier_orders:
            store_orders[order["store_id"]].append(order)
        
        # Create consolidated orders
        consolidated_orders = []
        
        for store_id, orders in store_orders.items():
            if len(orders) >= self.consolidation_threshold:
                # Create a consolidated order
                total_cost = sum(order["total_cost"] for order in orders)
                order_ids = [order["order_id"] for order in orders]
                
                consolidated_id = f"CONS-{supplier_id}-{store_id}-{int(datetime.now().timestamp())}"
                
                consolidated_order = {
                    "consolidated_id": consolidated_id,
                    "supplier_id": supplier_id,
                    "store_id": store_id,
                    "order_ids": order_ids,
                    "total_cost": total_cost,
                    "order_count": len(orders),
                    "created_at": datetime.now().isoformat()
                }
                
                consolidated_orders.append(consolidated_order)
                
                # Update status of individual orders
                for order in orders:
                    self.update_order_status(
                        order["order_id"], 
                        "consolidated",
                        {"consolidated_id": consolidated_id}
                    )
                
                # Publish consolidation notification
                self._publish_order_consolidation(consolidated_order)
        
        return consolidated_orders
    
    def _publish_order_consolidation(self, consolidated_order: Dict):
        """
        Publish order consolidation notification
        
        Args:
            consolidated_order: Consolidated order details
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "orders_consolidated",
            "consolidated_id": consolidated_order["consolidated_id"],
            "supplier_id": consolidated_order["supplier_id"],
            "store_id": consolidated_order["store_id"],
            "order_count": consolidated_order["order_count"],
            "total_cost": consolidated_order["total_cost"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("supply_chain_updates", message)
    
    def expedite_order(self, order_id: str) -> bool:
        """
        Expedite an existing order
        
        Args:
            order_id: The order identifier
            
        Returns:
            Success status
        """
        if order_id not in self.pending_orders:
            return False
            
        order = self.pending_orders[order_id]
        
        # Check if already expedited
        if order.get("is_expedited", False):
            return True
            
        # Check if order can be expedited (only if it's still in created state)
        if order["status"] != "created":
            return False
            
        # Update order to expedited
        additional_info = {
            "is_expedited": True,
            "total_cost": order["total_cost"] * self.expedite_cost_multiplier
        }
        
        # Recalculate expected delivery date
        current_lead_time = (datetime.fromisoformat(order["expected_delivery_date"]) - 
                           datetime.fromisoformat(order["order_date"])).days
        
        new_lead_time = max(1, round(current_lead_time * 0.7))
        new_delivery_date = (datetime.fromisoformat(order["order_date"]) + 
                           timedelta(days=new_lead_time)).isoformat()
        
        additional_info["expected_delivery_date"] = new_delivery_date
        additional_info["expected_lead_time"] = new_lead_time
        
        # Update the order
        success = self.update_order_status(order_id, "expedited", additional_info)
        
        if success:
            # Publish expedite notification
            self._publish_order_expedite(self.pending_orders[order_id])
            
        return success
    
    def _publish_order_expedite(self, order: Dict):
        """
        Publish order expedite notification
        
        Args:
            order: Expedited order details
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "order_expedited",
            "order_id": order["order_id"],
            "product_id": order["product_id"],
            "store_id": order["store_id"],
            "new_delivery_date": order["expected_delivery_date"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("supply_chain_updates", message)
    
    def check_for_potential_issues(self) -> List[Dict]:
        """
        Check for potential supply chain issues
        
        Returns:
            List of identified issues
        """
        issues = []
        
        # Check for supplier performance issues
        for supplier_id in range(1, 6):  # Assuming 5 suppliers in our demo
            performance = self.check_supplier_performance(supplier_id)
            
            # Flag suppliers with poor performance
            if performance["overall_score"] < 0.85:
                issue = {
                    "type": "supplier_performance",
                    "supplier_id": supplier_id,
                    "supplier_name": performance["supplier_name"],
                    "overall_score": performance["overall_score"],
                    "severity": "medium" if performance["overall_score"] > 0.8 else "high",
                    "identified_at": datetime.now().isoformat()
                }
                
                issues.append(issue)
                
                # Publish supplier issue notification
                self._publish_supplier_issue(issue)
        
        # Check for delayed orders
        today = datetime.now()
        
        for order_id, order in self.pending_orders.items():
            # Skip orders that are not in transit
            if order["status"] not in ["created", "expedited", "in_transit"]:
                continue
                
            # Check if expected delivery date is in the past
            expected_delivery = datetime.fromisoformat(order["expected_delivery_date"])
            
            if expected_delivery < today:
                # This order is delayed
                days_delayed = (today - expected_delivery).days
                
                issue = {
                    "type": "delayed_order",
                    "order_id": order_id,
                    "product_id": order["product_id"],
                    "store_id": order["store_id"],
                    "supplier_id": order["supplier_id"],
                    "days_delayed": days_delayed,
                    "severity": "low" if days_delayed <= 1 else "medium" if days_delayed <= 3 else "high",
                    "identified_at": datetime.now().isoformat()
                }
                
                issues.append(issue)
                
                # Publish order delay notification
                self._publish_order_delay(issue)
        
        return issues
    
    def _publish_supplier_issue(self, issue: Dict):
        """
        Publish supplier issue notification
        
        Args:
            issue: Supplier issue details
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "supplier_issue",
            "supplier_id": issue["supplier_id"],
            "supplier_name": issue["supplier_name"],
            "issue_type": "performance",
            "severity": issue["severity"],
            "score": issue["overall_score"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("supplier_alerts", message)
    
    def _publish_order_delay(self, issue: Dict):
        """
        Publish order delay notification
        
        Args:
            issue: Order delay details
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "order_delay",
            "order_id": issue["order_id"],
            "product_id": issue["product_id"],
            "store_id": issue["store_id"],
            "supplier_id": issue["supplier_id"],
            "days_delayed": issue["days_delayed"],
            "severity": issue["severity"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("supply_chain_alerts", message)
    
    def recommend_stock_transfer(self, product_id: int, destination_store_id: int, 
                              required_quantity: int) -> Dict:
        """
        Recommend stock transfer between stores for a product
        
        Args:
            product_id: The product identifier
            destination_store_id: Destination store ID
            required_quantity: Quantity needed
            
        Returns:
            Transfer recommendation
        """
        # Get inventory levels at all stores for this product
        stores_inventory = self._get_product_inventory_across_stores(product_id)
        
        # Remove destination store
        if destination_store_id in stores_inventory:
            del stores_inventory[destination_store_id]
            
        if not stores_inventory:
            return {"success": False, "reason": "No other stores with this product"}
            
        # Find stores with excess inventory
        transfer_candidates = []
        
        for store_id, inventory in stores_inventory.items():
            stock_level = inventory.get("stock_level", 0)
            reorder_point = inventory.get("reorder_point", 0)
            
            # Check if store has excess inventory
            if stock_level > reorder_point * 1.5:
                # Calculate transferable quantity
                transferable = stock_level - int(reorder_point * 1.2)  # Keep some buffer
                
                if transferable > 0:
                    transfer_candidates.append({
                        "source_store_id": store_id,
                        "transferable_quantity": transferable,
                        "distance": self._calculate_store_distance(store_id, destination_store_id)
                    })
        
        if not transfer_candidates:
            return {"success": False, "reason": "No stores with excess inventory"}
            
        # Sort by distance (closest first) and transferable quantity (highest first)
        transfer_candidates.sort(key=lambda x: (x["distance"], -x["transferable_quantity"]))
        
        # Create transfer plan
        transfer_plan = []
        remaining_quantity = required_quantity
        
        for candidate in transfer_candidates:
            if remaining_quantity <= 0:
                break
                
            transfer_quantity = min(candidate["transferable_quantity"], remaining_quantity)
            
            transfer_plan.append({
                "source_store_id": candidate["source_store_id"],
                "quantity": transfer_quantity,
                "distance": candidate["distance"]
            })
            
            remaining_quantity -= transfer_quantity
        
        # Check if we can fulfill the entire requirement
        success = remaining_quantity <= 0
        
        # Use LLM to enhance recommendation with reasoning
        enhanced_recommendation = self._enhance_transfer_recommendation(
            product_id, destination_store_id, required_quantity,
            transfer_plan, success, remaining_quantity
        )
        
        return enhanced_recommendation
    
    def _get_product_inventory_across_stores(self, product_id: int) -> Dict:
        """
        Get inventory levels for a product across all stores
        
        Args:
            product_id: The product identifier
            
        Returns:
            Dictionary mapping store IDs to inventory data
        """
        query = """
        SELECT 
            Store_ID, 
            Stock_Levels, 
            Reorder_Point, 
            Supplier_Lead_Time_days
        FROM 
            inventory_monitoring
        WHERE 
            Product_ID = ?
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, (product_id,))
        results = cursor.fetchall()
        
        inventory_by_store = {}
        
        for row in results:
            store_id, stock_level, reorder_point, lead_time = row
            
            inventory_by_store[store_id] = {
                "stock_level": stock_level,
                "reorder_point": reorder_point,
                "lead_time": lead_time
            }
        
        return inventory_by_store
    
    def _calculate_store_distance(self, store_id1: int, store_id2: int) -> float:
        """
        Calculate distance between two stores
        
        Args:
            store_id1: First store ID
            store_id2: Second store ID
            
        Returns:
            Distance value (higher means further apart)
        """
        # In a real implementation, this would use geocoding and actual distances
        # For this demo, we'll simulate using store ID differences
        return abs(store_id1 - store_id2)
    
    def _enhance_transfer_recommendation(self, product_id: int, destination_store_id: int,
                                       required_quantity: int, transfer_plan: List[Dict],
                                       success: bool, remaining_quantity: int) -> Dict:
        """
        Enhance stock transfer recommendation with LLM reasoning
        
        Args:
            product_id: The product identifier
            destination_store_id: Destination store ID
            required_quantity: Quantity needed
            transfer_plan: Proposed transfer plan
            success: Whether plan fulfills entire requirement
            remaining_quantity: Quantity still unfulfilled
            
        Returns:
            Enhanced recommendation
        """
        # Get product information
        product_info = self.get_product_info(product_id)
        
        # Format transfer plan for prompt
        transfer_text = ""
        total_transfer = 0
        
        for transfer in transfer_plan:
            transfer_text += (f"- From Store {transfer['source_store_id']}: "
                             f"{transfer['quantity']} units "
                             f"(distance factor: {transfer['distance']})\n")
            total_transfer += transfer["quantity"]
        
        # Create prompt for LLM
        prompt = f"""
        You are a supply chain optimization expert. I need your analysis on the following stock transfer recommendation:
        
        Product: ID {product_id}, Category: {product_info.get('category', 'unknown')}
        Destination: Store {destination_store_id}
        Required quantity: {required_quantity} units
        
        Proposed transfer plan:
        {transfer_text}
        
        Total transfer quantity: {total_transfer} units
        {'Plan covers entire requirement' if success else f'Plan is short by {remaining_quantity} units'}
        
        Please analyze this transfer plan and provide:
        1. Whether this plan should be executed or if we should order from suppliers instead
        2. Any risks or considerations for this transfer
        3. Implementation recommendations (timing, prioritization, etc.)
        
        Format your response as follows:
        RECOMMENDATION: [one of: EXECUTE_TRANSFER, PARTIAL_TRANSFER, ORDER_FROM_SUPPLIER]
        
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
        parsed_response = self._parse_transfer_recommendation(llm_response)
        
        # Combine information
        final_recommendation = {
            "product_id": product_id,
            "destination_store_id": destination_store_id,
            "required_quantity": required_quantity,
            "transfer_plan": transfer_plan,
            "total_transfer_quantity": total_transfer,
            "recommendation_type": parsed_response["recommendation_type"],
            "justification": parsed_response["justification"],
            "risks": parsed_response["risks"],
            "implementation": parsed_response["implementation"],
            "success": success,
            "remaining_quantity": remaining_quantity if not success else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        return final_recommendation
    
    def _parse_transfer_recommendation(self, llm_response: str) -> Dict:
        """
        Parse LLM transfer recommendation response
        
        Args:
            llm_response: LLM response text
            
        Returns:
            Parsed response dictionary
        """
        recommendation_type = "EXECUTE_TRANSFER"
        justification = ""
        risks = []
        implementation = ""
        
        # Extract sections
        sections = {
            "RECOMMENDATION": "",
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
            if line.startswith("RECOMMENDATION:"):
                current_section = "RECOMMENDATION"
                sections[current_section] = line.replace("RECOMMENDATION:", "").strip()
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
        rec_type = sections["RECOMMENDATION"].upper()
        if "PARTIAL_TRANSFER" in rec_type:
            recommendation_type = "PARTIAL_TRANSFER"
        elif "ORDER_FROM_SUPPLIER" in rec_type:
            recommendation_type = "ORDER_FROM_SUPPLIER"
        else:
            recommendation_type = "EXECUTE_TRANSFER"
        
        # Process other sections
        justification = sections["JUSTIFICATION"]
        risks = sections["RISKS"]
        implementation = sections["IMPLEMENTATION"]
        
        return {
            "recommendation_type": recommendation_type,
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
    
    def handle_inventory_command(self, message: Dict):
        """
        Handle inventory commands from other agents
        
        Args:
            message: Message containing inventory command
        """
        command_type = message.get("type")
        
        if command_type == "reorder":
            # Handle reorder command
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            quantity = message.get("quantity")
            is_expedited = message.get("expedite", False)
            
            if product_id and store_id and quantity:
                self.create_order(product_id, store_id, quantity, is_expedited)
                
        elif command_type == "expedite_order":
            # Handle expedite command
            order_id = message.get("order_id")
            
            if order_id:
                self.expedite_order(order_id)
                
        elif command_type == "initiate_transfer":
            # Handle stock transfer command
            product_id = message.get("product_id")
            destination_store_id = message.get("destination_store_id")
            required_quantity = message.get("required_quantity")
            
            if product_id and destination_store_id:
                # If quantity not specified, use a default
                if not required_quantity:
                    required_quantity = 10
                    
                recommendation = self.recommend_stock_transfer(
                    product_id, destination_store_id, required_quantity
                )
                
                # Publish transfer recommendation
                self._publish_transfer_recommendation(recommendation)
    
    def _publish_transfer_recommendation(self, recommendation: Dict):
        """
        Publish stock transfer recommendation
        
        Args:
            recommendation: Transfer recommendation
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "transfer_recommendation",
            "product_id": recommendation["product_id"],
            "destination_store_id": recommendation["destination_store_id"],
            "transfer_plan": recommendation["transfer_plan"],
            "total_quantity": recommendation["total_transfer_quantity"],
            "recommendation_type": recommendation["recommendation_type"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("supply_chain_recommendations", message)
    
    def handle_approved_recommendation(self, message: Dict):
        """
        Handle approved recommendations from the coordinator
        
        Args:
            message: Message containing approved recommendation
        """
        # Only handle inventory recommendations
        if message.get("type") != "inventory":
            return
            
        status = message.get("status")
        
        if status == "approved":
            # Approved as is
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            data = message.get("data", {})
            
            if "quantity" in data:
                # Create order
                is_expedited = data.get("expedite", False)
                self.create_order(product_id, store_id, data["quantity"], is_expedited)
                
        elif status == "modified":
            # Handle modified recommendation
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            modifications = message.get("modifications", {})
            
            if "order_quantity" in modifications:
                # Create order with modified quantity
                is_expedited = modifications.get("expedite", False)
                self.create_order(product_id, store_id, modifications["order_quantity"], is_expedited)
    
    def handle_coordinated_action(self, message: Dict):
        """
        Handle coordinated actions from the coordinator
        
        Args:
            message: Message containing coordinated action
        """
        action_type = message.get("action_type")
        
        if action_type == "expedite_order":
            # Handle expedite command
            order_id = message.get("order_id")
            
            if order_id:
                self.expedite_order(order_id)
                
        elif action_type == "supplier_coordination":
            # Handle supplier coordination action
            supplier_id = message.get("supplier_id")
            
            if supplier_id:
                self.check_supplier_performance(supplier_id)
                
                # Consolidate orders for this supplier
                self.consolidate_orders(supplier_id)
    
    def run_periodic_supply_chain_check(self):
        """
        Run periodic supply chain checks
        """
        # Check for supply chain issues
        issues = self.check_for_potential_issues()
        
        # Try to consolidate orders for each supplier
        consolidated = []
        
        for supplier_id in range(1, 6):  # Assuming 5 suppliers in our demo
            consolidated_orders = self.consolidate_orders(supplier_id)
            consolidated.extend(consolidated_orders)
        
        # Update any simulated order statuses
        # In a real implementation, this would check with external systems
        self._simulate_order_progress()
        
        return {
            "issues_detected": len(issues),
            "orders_consolidated": len(consolidated),
            "timestamp": datetime.now().isoformat()
        }
    
    def _simulate_order_progress(self):
        """
        Simulate progress of orders over time
        This is only for demonstration purposes
        """
        today = datetime.now()
        
        for order_id, order in list(self.pending_orders.items()):
            # Skip orders that are already delivered or cancelled
            if order["status"] in ["delivered", "cancelled"]:
                continue
                
            # Calculate days since order creation
            order_date = datetime.fromisoformat(order["order_date"])
            days_since_order = (today - order_date).days
            
            # Simulate order progress based on expected lead time
            expected_lead_time = order.get("expected_lead_time", 7)
            
            if order["status"] == "created" and days_since_order >= 1:
                # Move to "in_transit" after 1 day
                self.update_order_status(order_id, "in_transit")
                
            elif order["status"] == "expedited" and days_since_order >= 1:
                # Move expedited orders to "in_transit" after 1 day
                self.update_order_status(order_id, "in_transit")
                
            elif order["status"] == "in_transit":
                if days_since_order >= expected_lead_time:
                    # Order has arrived
                    self.update_order_status(order_id, "delivered")
                    
                    # Publish delivery notification
                    self._publish_order_delivery(self.pending_orders[order_id])
    
    def _publish_order_delivery(self, order: Dict):
        """
        Publish order delivery notification
        
        Args:
            order: Delivered order details
        """
        if not self.message_bus:
            return
            
        message = {
            "type": "order_delivered",
            "order_id": order["order_id"],
            "product_id": order["product_id"],
            "store_id": order["store_id"],
            "quantity": order["quantity"],
            "delivery_date": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_bus.publish("supply_chain_updates", message)