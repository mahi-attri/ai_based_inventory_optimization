#!/usr/bin/env python3
# frontend/app.py - Web server for Retail Inventory Optimizer Dashboard

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import sqlite3
from flask import Flask, render_template, jsonify, request, send_from_directory

# Add parent directory to path so we can import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from main project
from database_init import initialize_database
from agents.demand_agent import DemandForecastingAgent
from agents.inventory_agent import InventoryAgent
from agents.pricing_agent import PricingOptimizationAgent
from agents.supply_chain_agent import SupplyChainAgent
from agents.customer_agent import CustomerBehaviorAgent
from agents.coordinator_agent import CoordinationAgent
from agents.reporting_agent import ReportingAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("frontend")

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Global variables to store agent instances and configuration
agents = {}
config = {}
db_path = ""

def load_config(config_path="config/config.json"):
    """Load system configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Return a default config if the file can't be loaded
        return {
            "system": {
                "run_mode": "continuous",
                "run_interval_seconds": 300,
                "ollama_base_url": "http://localhost:11434"
            }
        }

def init_agents(config, db_path):
    """Initialize all agents in the system"""
    # Get Ollama base URL from config
    ollama_base_url = config.get("system", {}).get("ollama_base_url", "http://localhost:11434")
    
    # Initialize coordinator agent first
    coordinator = CoordinationAgent(
        config.get("coordinator_agent", {}),
        db_path,
        ollama_base_url
    )
    
    # Initialize specialized agents
    demand_agent = DemandForecastingAgent(
        config.get("demand_agent_config", {}),
        db_path,
        ollama_base_url
    )
    
    inventory_agent = InventoryAgent(
        config.get("inventory_agent_config", {}),
        db_path,
        ollama_base_url
    )
    
    pricing_agent = PricingOptimizationAgent(
        config.get("pricing_agent_config", {}),
        db_path,
        ollama_base_url
    )
    
    supply_chain_agent = SupplyChainAgent(
        config.get("supply_chain_agent_config", {}),
        db_path,
        ollama_base_url
    )
    
    customer_agent = CustomerBehaviorAgent(
        config.get("customer_agent_config", {}),
        db_path,
        ollama_base_url
    )
    
    reporting_agent = ReportingAgent(
        config.get("reporting_agent_config", {}),
        db_path,
        ollama_base_url
    )
    
    # Register all agents with coordinator
    coordinator.register_agent("demand_agent", demand_agent)
    coordinator.register_agent("inventory_agent", inventory_agent)
    coordinator.register_agent("pricing_agent", pricing_agent)
    coordinator.register_agent("supply_chain_agent", supply_chain_agent)
    coordinator.register_agent("customer_agent", customer_agent)
    coordinator.register_agent("reporting_agent", reporting_agent)
    
    # Initialize the coordination system
    coordinator.initialize_system()
    
    # Return dictionary of all agents
    return {
        "coordinator": coordinator,
        "demand_agent": demand_agent,
        "inventory_agent": inventory_agent,
        "pricing_agent": pricing_agent,
        "supply_chain_agent": supply_chain_agent,
        "customer_agent": customer_agent,
        "reporting_agent": reporting_agent
    }

def get_recent_logs(limit=5):
    """Get recent log entries from the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, agent_id, action, status 
            FROM agent_activity_log 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        logs = [
            {
                "time": log[0],
                "agent": log[1],
                "activity": log[2],
                "status": log[3]
            } 
            for log in cursor.fetchall()
        ]
        
        conn.close()
        return logs
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        # Return mock data if database query fails
        return [
            {
                "time": (datetime.now() - timedelta(minutes=i*5)).strftime("%H:%M %p"),
                "agent": agent,
                "activity": f"Periodic {activity}",
                "status": "Success"
            }
            for i, (agent, activity) in enumerate([
                ("Reporting Agent", "report generation"),
                ("Inventory Agent", "inventory check"),
                ("Coordinator", "coordination task"),
                ("Pricing Agent", "price optimization"),
                ("Demand Agent", "demand forecast")
            ])
        ]

def get_inventory_alerts(limit=4):
    """Get current inventory alerts from the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                p.name, 
                c.name, 
                s.name, 
                i.status,
                i.current_stock,
                i.recommended_stock
            FROM inventory_alerts i
            JOIN products p ON i.product_id = p.id
            JOIN categories c ON p.category_id = c.id
            JOIN stores s ON i.store_id = s.id
            ORDER BY 
                CASE i.status
                    WHEN 'Out of Stock' THEN 1
                    WHEN 'Low Stock' THEN 2
                    WHEN 'Overstocked' THEN 3
                    ELSE 4
                END
            LIMIT ?
        """, (limit,))
        
        alerts = [
            {
                "product": alert[0],
                "category": alert[1],
                "store": alert[2],
                "status": alert[3],
                "current_stock": alert[4],
                "recommended": alert[5]
            }
            for alert in cursor.fetchall()
        ]
        
        conn.close()
        return alerts
    except Exception as e:
        logger.error(f"Error retrieving inventory alerts: {e}")
        # Return mock data if database query fails
        return [
            {
                "product": "Organic Bananas",
                "category": "Groceries",
                "store": "Store #152",
                "status": "Out of Stock",
                "current_stock": 0,
                "recommended": "120 units"
            },
            {
                "product": "Wireless Earbuds",
                "category": "Electronics",
                "store": "Store #103",
                "status": "Low Stock",
                "current_stock": 3,
                "recommended": "15 units"
            },
            {
                "product": "Winter Jackets",
                "category": "Clothing",
                "store": "Store #078",
                "status": "Overstocked",
                "current_stock": 87,
                "recommended": "30 units"
            },
            {
                "product": "Coffee Maker",
                "category": "Appliances",
                "store": "Store #042",
                "status": "Optimal",
                "current_stock": 12,
                "recommended": "10-15 units"
            }
        ]

def get_dashboard_summary():
    """Get summary statistics for the dashboard"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get out of stock count
        cursor.execute("""
            SELECT COUNT(*) FROM inventory_alerts
            WHERE status = 'Out of Stock'
        """)
        out_of_stock = cursor.fetchone()[0]
        
        # Get overstocked count
        cursor.execute("""
            SELECT COUNT(*) FROM inventory_alerts
            WHERE status = 'Overstocked'
        """)
        overstocked = cursor.fetchone()[0]
        
        # Get weekly lost sales
        cursor.execute("""
            SELECT SUM(lost_sales_value) FROM lost_sales
            WHERE date >= date('now', '-7 days')
        """)
        weekly_lost_sales = cursor.fetchone()[0] or 0
        
        conn.close()
        
        # Get percent changes
        out_of_stock_change = -12  # Mock data
        overstocked_change = -8    # Mock data
        weekly_lost_sales_change = 5  # Mock data
        
        return {
            "out_of_stock": {
                "value": out_of_stock,
                "change": out_of_stock_change
            },
            "overstocked": {
                "value": overstocked,
                "change": overstocked_change
            },
            "weekly_lost_sales": {
                "value": weekly_lost_sales,
                "change": weekly_lost_sales_change
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving dashboard summary: {e}")
        # Return mock data if database query fails
        return {
            "out_of_stock": {
                "value": 24,
                "change": -12
            },
            "overstocked": {
                "value": 43,
                "change": -8
            },
            "weekly_lost_sales": {
                "value": 12350,
                "change": 5
            }
        }

def get_agent_status():
    """Get status information for all agents"""
    agent_data = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # For each agent, get the latest activity
        for agent_id, agent in agents.items():
            cursor.execute("""
                SELECT 
                    timestamp,
                    status,
                    metrics
                FROM agent_activity_log
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (agent_id,))
            
            result = cursor.fetchone()
            
            if result:
                last_run = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S.%f")
                status = result[1]
                metrics = json.loads(result[2]) if result[2] else {}
                
                # Calculate time since last run
                now = datetime.now()
                diff = now - last_run
                
                if diff.total_seconds() < 60:
                    time_ago = f"{int(diff.total_seconds())} seconds ago"
                elif diff.total_seconds() < 3600:
                    time_ago = f"{int(diff.total_seconds() / 60)} minutes ago"
                else:
                    time_ago = f"{int(diff.total_seconds() / 3600)} hours ago"
                
                # Determine if agent is active
                is_active = diff.total_seconds() < config.get("system", {}).get("run_interval_seconds", 300) * 2
                
                agent_data[agent_id] = {
                    "name": agent_id.replace("_agent", "").replace("_", " ").title(),
                    "status": "active" if is_active and status == "success" else "idle" if status == "success" else "error",
                    "last_run": time_ago,
                    "metrics": metrics
                }
            else:
                # No activity found
                agent_data[agent_id] = {
                    "name": agent_id.replace("_agent", "").replace("_", " ").title(),
                    "status": "offline",
                    "last_run": "Never",
                    "metrics": {}
                }
                
        conn.close()
    except Exception as e:
        logger.error(f"Error retrieving agent status: {e}")
        # Return mock data for agent status
        agent_data = {
            "demand_agent": {
                "name": "Demand Forecasting Agent",
                "status": "active",
                "last_run": "10 minutes ago",
                "metrics": {
                    "accuracy": "92%",
                    "products_analyzed": 1245
                }
            },
            "inventory_agent": {
                "name": "Inventory Agent",
                "status": "active",
                "last_run": "3 minutes ago",
                "metrics": {
                    "alerts_generated": 24,
                    "orders_recommended": 18
                }
            },
            "pricing_agent": {
                "name": "Pricing Optimization Agent",
                "status": "idle",
                "last_run": "2 hours ago",
                "metrics": {
                    "price_changes": 43,
                    "revenue_impact": "+$5,230"
                }
            },
            "supply_chain_agent": {
                "name": "Supply Chain Agent",
                "status": "active",
                "last_run": "15 minutes ago",
                "metrics": {
                    "shipments_tracked": 32,
                    "delivery_issues": 3
                }
            },
            "customer_agent": {
                "name": "Customer Behavior Agent",
                "status": "active",
                "last_run": "30 minutes ago",
                "metrics": {
                    "segments_identified": 8,
                    "insights_generated": 12
                }
            },
            "reporting_agent": {
                "name": "Reporting Agent",
                "status": "active",
                "last_run": "5 minutes ago",
                "metrics": {
                    "reports_generated": 4,
                    "alert_notifications": 7
                }
            },
            "coordinator": {
                "name": "Coordination Agent",
                "status": "active",
                "last_run": "2 minutes ago",
                "metrics": {
                    "tasks_coordinated": 36,
                    "recommendations": 15
                }
            }
        }
        
    return agent_data

def get_inventory_health_data(period="week"):
    """Get inventory health data for chart"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get inventory status by category
        cursor.execute("""
            SELECT 
                c.name as category,
                i.status,
                COUNT(*) as count
            FROM inventory_alerts i
            JOIN products p ON i.product_id = p.id
            JOIN categories c ON p.category_id = c.id
            GROUP BY c.name, i.status
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        # Process results into chart data format
        categories = []
        optimal_data = []
        low_stock_data = []
        out_of_stock_data = []
        overstocked_data = []
        
        # Create a dictionary to organize the data
        data_by_category = {}
        
        for row in results:
            category, status, count = row
            
            if category not in data_by_category:
                data_by_category[category] = {
                    "Optimal": 0,
                    "Low Stock": 0,
                    "Out of Stock": 0,
                    "Overstocked": 0
                }
                
            data_by_category[category][status] = count
        
        # Convert to chart format
        for category, stats in data_by_category.items():
            categories.append(category)
            optimal_data.append(stats["Optimal"])
            low_stock_data.append(stats["Low Stock"])
            out_of_stock_data.append(stats["Out of Stock"])
            overstocked_data.append(stats["Overstocked"])
        
        return {
            "labels": categories,
            "datasets": [
                {
                    "label": "Optimal Stock",
                    "data": optimal_data,
                    "backgroundColor": "rgba(16, 185, 129, 0.6)"
                },
                {
                    "label": "Low Stock",
                    "data": low_stock_data,
                    "backgroundColor": "rgba(245, 158, 11, 0.6)"
                },
                {
                    "label": "Out of Stock",
                    "data": out_of_stock_data,
                    "backgroundColor": "rgba(239, 68, 68, 0.6)"
                },
                {
                    "label": "Overstocked",
                    "data": overstocked_data,
                    "backgroundColor": "rgba(37, 99, 235, 0.6)"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving inventory health data: {e}")
        # Return mock data if database query fails
        return {
            "labels": ['Groceries', 'Electronics', 'Clothing', 'Home Goods', 'Toys', 'Beauty'],
            "datasets": [
                {
                    "label": "Optimal Stock",
                    "data": [65, 42, 73, 56, 38, 62],
                    "backgroundColor": "rgba(16, 185, 129, 0.6)"
                },
                {
                    "label": "Low Stock",
                    "data": [12, 8, 5, 10, 7, 4],
                    "backgroundColor": "rgba(245, 158, 11, 0.6)"
                },
                {
                    "label": "Out of Stock",
                    "data": [3, 5, 2, 4, 1, 3],
                    "backgroundColor": "rgba(239, 68, 68, 0.6)"
                },
                {
                    "label": "Overstocked",
                    "data": [8, 12, 15, 6, 9, 5],
                    "backgroundColor": "rgba(37, 99, 235, 0.6)"
                }
            ]
        }

# Route handlers
@app.route('/')
def index():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/api/dashboard/summary')
def dashboard_summary():
    """API endpoint for dashboard summary data"""
    return jsonify(get_dashboard_summary())

@app.route('/api/agents/status')
def agent_status():
    """API endpoint for agent status"""
    return jsonify(get_agent_status())

@app.route('/api/inventory/alerts')
def inventory_alerts():
    """API endpoint for inventory alerts"""
    limit = request.args.get('limit', 4, type=int)
    return jsonify(get_inventory_alerts(limit))

@app.route('/api/inventory/health')
def inventory_health():
    """API endpoint for inventory health data"""
    period = request.args.get('period', 'week')
    return jsonify(get_inventory_health_data(period))

@app.route('/api/activity/recent')
def recent_activity():
    """API endpoint for recent system activity"""
    limit = request.args.get('limit', 5, type=int)
    return jsonify(get_recent_logs(limit))

@app.route('/api/config')
def get_config():
    """API endpoint to get current configuration"""
    return jsonify(config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """API endpoint to update configuration"""
    new_config = request.json
    
    # Merge new config with existing config
    for key, value in new_config.items():
        if key in config:
            if isinstance(value, dict) and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
        else:
            config[key] = value
    
    # Save updated config to file
    try:
        with open('config/config.json', 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    return jsonify({"status": "success", "config": config})

@app.route('/api/agents/run', methods=['POST'])
def run_agents():
    """API endpoint to run agents"""
    run_data = request.json
    agent_ids = run_data.get('agents', [])
    run_mode = run_data.get('mode', 'single')
    
    results = {}
    
    try:
        # If no specific agents selected, run all
        if not agent_ids or 'all' in agent_ids:
            agent_ids = list(agents.keys())
        
        # Run each selected agent
        for agent_id in agent_ids:
            if agent_id in agents:
                agent = agents[agent_id]
                
                # Call the appropriate run method based on agent type
                if agent_id == "demand_agent":
                    result = agent.run_periodic_demand_forecast()
                elif agent_id == "inventory_agent":
                    result = agent.run_periodic_inventory_check()
                elif agent_id == "pricing_agent":
                    result = agent.run_periodic_pricing_optimization()
                elif agent_id == "supply_chain_agent":
                    result = agent.run_periodic_supply_chain_check()
                elif agent_id == "customer_agent":
                    result = agent.run_periodic_customer_analysis()
                elif agent_id == "reporting_agent":
                    result = agent.run_periodic_reporting()
                elif agent_id == "coordinator":
                    result = agent.run_periodic_coordination()
                else:
                    result = {"status": "error", "message": f"Unknown run method for agent {agent_id}"}
                
                results[agent_id] = result
            else:
                results[agent_id] = {"status": "error", "message": f"Agent {agent_id} not found"}
        
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        logger.error(f"Error running agents: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Initialize the application
def initialize_app():
    global config, agents, db_path
    
    # Load configuration
    config = load_config()
    
    # Set database path
    db_path = os.environ.get('DB_PATH', 'data/retail_inventory.db')
    
    # Initialize database if it doesn't exist
    if not os.path.exists(db_path):
        logger.info(f"Initializing database at {db_path}")
        initialize_database(db_path)
    
    # Initialize agents
    agents = init_agents(config, db_path)
    
    logger.info("Frontend application initialized")

if __name__ == '__main__':
    # Initialize the app before running
    initialize_app()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = True  # Set to True for more output
    
    print(f"Starting frontend server on port {port}")  # Add this line
    logger.info(f"Starting frontend server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)