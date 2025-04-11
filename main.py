# main.py
import argparse
import os
import logging
import sqlite3
from datetime import datetime
import time
import json
from typing import Dict

# Import database initialization function
from database_init import initialize_database

# Import agent modules
from agents.demand_agent import DemandForecastingAgent
from agents.inventory_agent import InventoryAgent
from agents.pricing_agent import PricingOptimizationAgent
from agents.supply_chain_agent import SupplyChainAgent
from agents.customer_agent import CustomerBehaviorAgent
from agents.coordinator_agent import CoordinationAgent
from agents.reporting_agent import ReportingAgent  # New import

def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def setup_logging(log_path: str, level=logging.INFO):
    """
    Set up logging configuration
    
    Args:
        log_path: Path to log file
        level: Logging level
    """
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Set up logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # Get root logger
    logger = logging.getLogger()
    
    return logger

def initialize_agents(config: Dict, db_path: str):
    """
    Initialize all agents in the system
    
    Args:
        config: Configuration dictionary
        db_path: Path to the SQLite database
        
    Returns:
        Dictionary of agent instances
    """
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
    
    # Initialize new reporting agent
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
    agents = {
        "coordinator": coordinator,
        "demand_agent": demand_agent,
        "inventory_agent": inventory_agent,
        "pricing_agent": pricing_agent,
        "supply_chain_agent": supply_chain_agent,
        "customer_agent": customer_agent,
        "reporting_agent": reporting_agent
    }
    
    return agents

def run_system(agents: Dict, config: Dict):
    """
    Run the multi-agent system
    
    Args:
        agents: Dictionary of agent instances
        config: Configuration dictionary
    """
    logger = logging.getLogger("main")
    
    # Get run configuration
    run_mode = config.get("system", {}).get("run_mode", "continuous")
    run_interval = config.get("system", {}).get("run_interval_seconds", 300)  # 5 minutes default
    
    logger.info(f"Starting system in {run_mode} mode")
    
    try:
        if run_mode == "continuous":
            # Run continuously with periodic checks
            while True:
                # Perform periodic checks for each agent
                run_periodic_tasks(agents)
                
                # Sleep for the specified interval
                logger.debug(f"Sleeping for {run_interval} seconds")
                time.sleep(run_interval)
                
        elif run_mode == "single":
            # Run a single cycle
            run_periodic_tasks(agents)
            
        else:
            logger.error(f"Unknown run mode: {run_mode}")
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Error in main run loop: {e}")
        raise
    finally:
        # Perform cleanup
        logger.info("Performing cleanup")
        for agent_id, agent in agents.items():
            if hasattr(agent, "cleanup"):
                agent.cleanup()

def run_periodic_tasks(agents: Dict):
    """
    Run periodic tasks for all agents
    
    Args:
        agents: Dictionary of agent instances
    """
    logger = logging.getLogger("main")
    
    # Record start time
    start_time = time.time()
    
    # Run coordinator periodic tasks
    logger.info("Running coordinator periodic tasks")
    coordinator_results = agents["coordinator"].run_periodic_coordination()
    
    # Run inventory agent periodic tasks
    logger.info("Running inventory agent periodic tasks")
    inventory_results = agents["inventory_agent"].run_periodic_inventory_check()
    
    # Run pricing agent periodic tasks
    logger.info("Running pricing agent periodic tasks")
    pricing_results = agents["pricing_agent"].run_periodic_pricing_optimization()
    
    # Run demand agent periodic tasks
    logger.info("Running demand agent periodic tasks")
    demand_results = agents["demand_agent"].run_periodic_demand_forecast()
    
    # Run supply chain agent periodic tasks
    logger.info("Running supply chain agent periodic tasks")
    supply_chain_results = agents["supply_chain_agent"].run_periodic_supply_chain_check()
    
    # Run customer agent periodic tasks
    logger.info("Running customer agent periodic tasks")
    customer_results = agents["customer_agent"].run_periodic_customer_analysis()
    
    # Run reporting agent periodic tasks
    logger.info("Running reporting agent periodic tasks")
    reporting_results = agents["reporting_agent"].run_periodic_reporting()
    
    # Calculate run duration
    duration = time.time() - start_time
    
    # Log summary
    logger.info(f"Completed periodic tasks in {duration:.2f} seconds")
    logger.info(f"Coordinator processed {coordinator_results.get('processed_recommendations', 0)} recommendations")
    
    return {
        "coordinator": coordinator_results,
        "inventory": inventory_results,
        "pricing": pricing_results,
        "demand": demand_results,
        "supply_chain": supply_chain_results,
        "customer": customer_results,
        "reporting": reporting_results,
        "duration": duration
    }

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Retail Inventory Multi-Agent System")
    
    parser.add_argument(
        "--config", 
        type=str,
        default="config/config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--db", 
        type=str,
        default="data/retail_inventory.db",
        help="Path to SQLite database"
    )
    
    parser.add_argument(
        "--log", 
        type=str,
        default="logs/retail_inventory.log",
        help="Path to log file"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--run-mode", 
        type=str,
        default="continuous",
        choices=["continuous", "single"],
        help="System run mode"
    )
    
    return parser.parse_args()

def main():
    """
    Main entry point for the application
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(args.log, level=log_level)
    
    logger.info("Starting Retail Inventory Multi-Agent System")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override config with command-line arguments
        config.setdefault("system", {})
        config["system"]["run_mode"] = args.run_mode
        
        # Initialize database
        logger.info(f"Initializing database at {args.db}")
        db_conn = initialize_database(args.db)
        
        # Initialize agents
        logger.info("Initializing agents")
        agents = initialize_agents(config, args.db)
        
        # Run the system
        logger.info("Starting system execution")
        run_system(agents, config)
        
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
        return 1
        
    logger.info("System execution completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)