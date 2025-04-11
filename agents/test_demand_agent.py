import json
import sqlite3
from agents.demand_agent import DemandForecastingAgent

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Initialize agent
demand_agent = DemandForecastingAgent(
    config.get("demand_agent_config", {}),
    "data/retail_inventory.db",
    "http://localhost:11434"
)

# Test forecasting for a product
forecast = demand_agent.generate_forecast(5000, 10)
print(json.dumps(forecast, indent=2))