import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional, Callable
import requests
import time
from utils.message_bus import MessageBus

class CoordinationAgent:
    """
    Agent responsible for coordinating communication and decision-making
    between specialized agents in the retail inventory system.
    """
    
    def __init__(self, config: Dict, db_path: str, ollama_base_url: str):
        """
        Initialize the Coordination Agent
        
        Args:
            config: Configuration dictionary for agent settings
            db_path: Path to SQLite database 
            ollama_base_url: URL for Ollama API
        """
        self.config = config
        self.db_conn = sqlite3.connect(db_path)
        self.ollama_url = ollama_base_url
        self.llm_model = config.get("llm_model", "llama3")
        
        # Initialize logging
        self.logger = logging.getLogger("coordinator_agent")
        
        # Initialize message bus
        self.message_bus = MessageBus(self.logger)
        
        # Initialize agent registry
        self.agents = {}
        
        # Initialize decision log
        self.decision_log = []
        
        # Initialize task queue
        self.task_queue = []
        
        # Initialize optimization thresholds
        self.demand_change_threshold = config.get("demand_change_threshold", 0.1)  # 10% demand change
        self.inventory_threshold = config.get("inventory_threshold", 0.2)  # 20% of reorder point
        self.price_change_threshold = config.get("price_change_threshold", 0.05)  # 5% price change
    
    def register_agent(self, agent_id: str, agent):
        """
        Register an agent with the coordinator
        
        Args:
            agent_id: Agent identifier
            agent: Agent instance
        """
        self.agents[agent_id] = agent
        
        # Connect agent to message bus
        agent.connect_to_message_bus(self.message_bus)
        
        self.logger.info(f"Registered agent: {agent_id}")
    
    def initialize_system(self):
        """
        Initialize the multi-agent system
        """
        try:
            # Register coordinator's own message handlers
            self.message_bus.subscribe("pricing_recommendations", self.handle_pricing_recommendation)
            self.message_bus.subscribe("inventory_recommendations", self.handle_inventory_recommendation)
            self.message_bus.subscribe("demand_alerts", self.handle_demand_alert)
            self.message_bus.subscribe("customer_insights", self.handle_customer_insight)
            
            # Setup is complete
            self.logger.info("Coordination agent initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing coordination agent: {e}")
    
    def handle_pricing_recommendation(self, message: Dict):
        """
        Handle pricing recommendation message
        
        Args:
            message: Pricing recommendation message
        """
        try:
            recommendation_type = message.get("recommendation_type")
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            current_price = message.get("current_price")
            recommended_price = message.get("recommended_price")
            
            if not all([product_id, store_id, current_price, recommended_price]):
                self.logger.warning(f"Incomplete pricing recommendation: {message}")
                return
            
            # Calculate price change percentage
            price_change_pct = (recommended_price - current_price) / current_price
            
            # Log the recommendation
            self.logger.info(f"Received pricing recommendation for product {product_id}, store {store_id}: " 
                           f"{current_price:.2f} → {recommended_price:.2f} ({price_change_pct:.1%})")
            
            # Add to decision log
            self.decision_log.append({
                "timestamp": datetime.now().isoformat(),
                "type": "pricing_recommendation",
                "product_id": product_id,
                "store_id": store_id,
                "current_price": current_price,
                "recommended_price": recommended_price,
                "price_change_pct": price_change_pct,
                "status": "pending"
            })
            
            # Process recommendation based on significance
            if abs(price_change_pct) >= self.price_change_threshold:
                # Schedule for processing
                self.task_queue.append({
                    "type": "update_price",
                    "priority": "medium" if abs(price_change_pct) < 0.1 else "high",
                    "data": {
                        "product_id": product_id,
                        "store_id": store_id,
                        "new_price": recommended_price,
                        "reason": message.get("reason", "Optimization")
                    },
                    "created_at": datetime.now().isoformat()
                })
            else:
                # Minor change, log but don't take action
                self.logger.info(f"Price change below threshold ({price_change_pct:.1%}), no action taken")
        
        except Exception as e:
            self.logger.error(f"Error in handle_pricing_recommendation: {e}")
    
    def handle_inventory_recommendation(self, message: Dict):
        """
        Handle inventory recommendation message
        
        Args:
            message: Inventory recommendation message
        """
        try:
            recommendations = message.get("recommendations", [])
            
            for rec in recommendations:
                rec_type = rec.get("type")
                product_id = rec.get("product_id")
                store_id = rec.get("store_id")
                
                if not product_id or not store_id:
                    continue
                
                # Log the recommendation
                self.logger.info(f"Received inventory recommendation ({rec_type}) for product {product_id}, store {store_id}")
                
                # Add to decision log
                self.decision_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": f"inventory_{rec_type}",
                    "product_id": product_id,
                    "store_id": store_id,
                    "details": rec,
                    "status": "pending"
                })
                
                # Process different recommendation types
                if rec_type == "reorder":
                    # Priority based on inventory level relative to reorder point
                    current_stock = rec.get("current_stock", 0)
                    reorder_point = rec.get("reorder_point", 0)
                    
                    if reorder_point > 0:
                        relative_stock = current_stock / reorder_point
                        
                        # Add to task queue
                        self.task_queue.append({
                            "type": "process_reorder",
                            "priority": "high" if relative_stock < 0.5 else "medium",
                            "data": rec,
                            "created_at": datetime.now().isoformat()
                        })
                
                elif rec_type == "expiry_alert":
                    days_to_expiry = rec.get("days_to_expiry", 0)
                    
                    # Add to task queue
                    self.task_queue.append({
                        "type": "process_expiry_alert",
                        "priority": "high" if days_to_expiry <= 3 else "medium",
                        "data": rec,
                        "created_at": datetime.now().isoformat()
                    })
        
        except Exception as e:
            self.logger.error(f"Error in handle_inventory_recommendation: {e}")
    
    def handle_demand_alert(self, message: Dict):
        """
        Handle demand alert message
        
        Args:
            message: Demand alert message
        """
        try:
            alert_type = message.get("type")
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            magnitude = message.get("magnitude", 0)
            
            if not product_id or not store_id:
                return
            
            # Log the alert
            self.logger.info(f"Received demand alert ({alert_type}) for product {product_id}, store {store_id}: {magnitude}% change")
            
            # Add to decision log
            self.decision_log.append({
                "timestamp": datetime.now().isoformat(),
                "type": f"demand_{alert_type}",
                "product_id": product_id,
                "store_id": store_id,
                "magnitude": magnitude,
                "details": message,
                "status": "pending"
            })
            
            # Process based on magnitude
            if magnitude >= self.demand_change_threshold * 100:  # Convert threshold to percentage
                # Add to task queue
                self.task_queue.append({
                    "type": f"process_{alert_type}",
                    "priority": "high" if magnitude >= 25 else "medium",
                    "data": message,
                    "created_at": datetime.now().isoformat()
                })
            else:
                self.logger.info(f"Demand change below threshold ({magnitude}%), no action taken")
        
        except Exception as e:
            self.logger.error(f"Error in handle_demand_alert: {e}")
    
    def handle_customer_insight(self, message: Dict):
        """
        Handle customer insight message
        
        Args:
            message: Customer insight message
        """
        try:
            insight_type = message.get("type")
            
            # Log the insight
            self.logger.info(f"Received customer insight: {insight_type}")
            
            # Add to decision log
            self.decision_log.append({
                "timestamp": datetime.now().isoformat(),
                "type": f"customer_{insight_type}",
                "details": message,
                "status": "pending"
            })
            
            # Process based on insight type
            if insight_type == "price_response_prediction":
                product_id = message.get("product_id")
                store_id = message.get("store_id")
                expected_demand_change_pct = message.get("expected_demand_change_pct", 0)
                
                # Only process significant demand changes
                if abs(expected_demand_change_pct) >= self.demand_change_threshold:
                    # Add to task queue
                    self.task_queue.append({
                        "type": "evaluate_price_impact",
                        "priority": "medium",
                        "data": message,
                        "created_at": datetime.now().isoformat()
                    })
            
            elif insight_type == "stockout_impact":
                product_id = message.get("product_id")
                store_id = message.get("store_id")
                loyalty_impact_score = message.get("loyalty_impact_score", 0)
                
                # Process high-impact stockouts
                if loyalty_impact_score >= 60:
                    # Add to task queue
                    self.task_queue.append({
                        "type": "address_stockout_impact",
                        "priority": "high",
                        "data": message,
                        "created_at": datetime.now().isoformat()
                    })
        
        except Exception as e:
            self.logger.error(f"Error in handle_customer_insight: {e}")
    
    def process_task_queue(self):
        """
        Process tasks in the task queue
        
        Returns:
            Number of tasks processed
        """
        try:
            # Sort tasks by priority and creation time
            self.task_queue.sort(key=lambda x: (
                0 if x["priority"] == "high" else 1 if x["priority"] == "medium" else 2,
                x["created_at"]
            ))
            
            processed_count = 0
            max_tasks = 10  # Process up to 10 tasks per cycle
            
            while self.task_queue and processed_count < max_tasks:
                task = self.task_queue.pop(0)
                task_type = task["type"]
                task_data = task["data"]
                
                self.logger.info(f"Processing task: {task_type} (priority: {task['priority']})")
                
                # Process based on task type
                if task_type == "update_price":
                    self._process_price_update(task_data)
                elif task_type == "process_reorder":
                    self._process_reorder(task_data)
                elif task_type == "process_expiry_alert":
                    self._process_expiry_alert(task_data)
                elif task_type == "process_demand_spike":
                    self._process_demand_spike(task_data)
                elif task_type == "process_demand_drop":
                    self._process_demand_drop(task_data)
                elif task_type == "evaluate_price_impact":
                    self._evaluate_price_impact(task_data)
                elif task_type == "address_stockout_impact":
                    self._address_stockout_impact(task_data)
                else:
                    self.logger.warning(f"Unknown task type: {task_type}")
                
                processed_count += 1
            
            return processed_count
        
        except Exception as e:
            self.logger.error(f"Error in process_task_queue: {e}")
            return 0
    
    def _process_price_update(self, data: Dict):
        """
        Process price update task
        
        Args:
            data: Task data
        """
        try:
            product_id = data["product_id"]
            store_id = data["store_id"]
            new_price = data["new_price"]
            reason = data.get("reason", "Optimization")
            
            # Get product information
            cursor = self.db_conn.cursor()
            cursor.execute("""
            SELECT p.Name, po.Price
            FROM products p
            JOIN pricing_optimization po ON p.Product_ID = po.Product_ID
            WHERE p.Product_ID = ? AND po.Store_ID = ?
            """, (product_id, store_id))
            
            result = cursor.fetchone()
            
            if not result:
                self.logger.warning(f"Product {product_id} not found in store {store_id}")
                return
            
            product_name, current_price = result
            
            # Fallback mechanism if no specific pricing agent
            try:
                # Update price in the database
                cursor.execute("""
                UPDATE pricing_optimization 
                SET Price = ?, Last_Updated = ?, Update_Reason = ?
                WHERE Product_ID = ? AND Store_ID = ?
                """, (
                    new_price, 
                    datetime.now().isoformat(), 
                    reason, 
                    product_id, 
                    store_id
                ))
                self.db_conn.commit()
                
                self.logger.info(f"Price updated for {product_name} (ID: {product_id}): {current_price:.2f} → {new_price:.2f}")
                
                # Update decision log
                for decision in self.decision_log:
                    if (decision["type"] == "pricing_recommendation" and 
                        decision["product_id"] == product_id and 
                        decision["store_id"] == store_id and
                        decision["status"] == "pending"):
                        decision["status"] = "implemented"
                        decision["implementation_time"] = datetime.now().isoformat()
            
            except sqlite3.Error as db_error:
                self.logger.error(f"Database error updating price: {db_error}")
                
            except Exception as update_error:
                self.logger.error(f"Error updating price: {update_error}")
        
        except Exception as e:
            self.logger.error(f"Unexpected error in _process_price_update: {e}")

    def run_periodic_coordination(self):
        """
        Run periodic coordination tasks across all agents
        
        Returns:
            Dictionary of coordination results
        """
        try:
            # Track processed recommendations
            processed_recommendations = 0
            
            # Process task queue
            processed_tasks = self.process_task_queue()
            
            # Additional coordination logic can be added here
            # For example, cross-agent recommendation synthesis
            
            # Log coordination activity
            self.logger.info(f"Processed {processed_tasks} tasks in periodic coordination")
            
            return {
                "processed_tasks": processed_tasks,
                "processed_recommendations": processed_recommendations,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Error in periodic coordination: {e}")
            return {
                "processed_tasks": 0,
                "processed_recommendations": 0,
                "error": str(e)
            }

    def cleanup(self):
        """
        Perform cleanup operations for the coordination agent
        """
        try:
            # Close database connection
            if hasattr(self, 'db_conn') and self.db_conn:
                self.db_conn.close()
            
            # Clear decision log
            self.decision_log.clear()
            
            # Clear task queue
            self.task_queue.clear()
            
            # Added additional placeholder methods for other agents if they exist
            for agent_id, agent in self.agents.items():
                if hasattr(agent, 'cleanup'):
                    try:
                        agent.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up agent {agent_id}: {e}")
            
            self.logger.info("Coordination Agent cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Error during Coordination Agent cleanup: {e}")

    # Optional: Placeholder methods for task processing
    def _process_reorder(self, data: Dict):
        """
        Process reorder task
        
        Args:
            data: Task data for reordering
        """
        self.logger.info(f"Processing reorder task: {data}")
    
    def _process_expiry_alert(self, data: Dict):
        """
        Process expiry alert task
        
        Args:
            data: Task data for expiry alert
        """
        self.logger.info(f"Processing expiry alert: {data}")
    
    def _process_demand_spike(self, data: Dict):
        """
        Process demand spike task
        
        Args:
            data: Task data for demand spike
        """
        self.logger.info(f"Processing demand spike: {data}")
    
    def _process_demand_drop(self, data: Dict):
        """
        Process demand drop task
        
        Args:
            data: Task data for demand drop
        """
        self.logger.info(f"Processing demand drop: {data}")
    
    def _evaluate_price_impact(self, data: Dict):
        """
        Evaluate price impact task
        
        Args:
            data: Task data for price impact evaluation
        """
        self.logger.info(f"Evaluating price impact: {data}")
    
    def _address_stockout_impact(self, data: Dict):
        """
        Address stockout impact task
        
        Args:
            data: Task data for stockout impact
        """
        self.logger.info(f"Addressing stockout impact: {data}")