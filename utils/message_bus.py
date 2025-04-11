import logging
from collections import defaultdict
from typing import Dict, Any, Callable, Optional

class MessageBus:
    """
    A comprehensive message bus for inter-agent communication in a multi-agent system.
    
    The MessageBus provides several key communication mechanisms:
    1. Publish-Subscribe (pub/sub) for event-driven communication
    2. Synchronous request-response for direct agent interactions
    3. Flexible logging and error handling
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the MessageBus
        
        Args:
            logger: Optional logger. If not provided, a default logger will be created.
        """
        # Subscribers for pub/sub communication
        self.subscribers = defaultdict(list)
        
        # Request handlers for synchronous communication
        self.request_handlers = {}
        
        # Use provided logger or create a default one
        self.logger = logger or logging.getLogger("message_bus")
    
    def subscribe(self, topic: str, callback: Callable):
        """
        Subscribe a callback function to a specific topic
        
        Args:
            topic: The topic to subscribe to
            callback: Function to be called when a message is published to the topic
        """
        self.subscribers[topic].append(callback)
        self.logger.debug(f"Subscribed to topic: {topic}")
    
    def publish(self, topic: str, message: Dict[str, Any]):
        """
        Publish a message to all subscribers of a topic
        
        Args:
            topic: The topic to publish to
            message: The message payload
        """
        self.logger.debug(f"Publishing to topic: {topic}")
        
        for callback in self.subscribers[topic]:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Error in subscriber callback for topic {topic}: {e}")
    
    def register_request_handler(self, agent_id: str, method: str, handler: Callable):
        """
        Register a handler for synchronous requests from a specific agent and method
        
        Args:
            agent_id: Identifier of the agent
            method: Method name for the request
            handler: Function to handle the request
        """
        key = f"{agent_id}.{method}"
        self.request_handlers[key] = handler
        self.logger.debug(f"Registered request handler: {key}")
    
    def request(self, agent_id: str, method: str, params: Dict[str, Any]) -> Any:
        """
        Send a synchronous request to a specific agent method
        
        Args:
            agent_id: Identifier of the target agent
            method: Method to call on the agent
            params: Parameters for the method
        
        Returns:
            Response from the agent or None if request fails
        """
        key = f"{agent_id}.{method}"
        
        if key not in self.request_handlers:
            self.logger.error(f"No handler registered for: {key}")
            return None
        
        try:
            return self.request_handlers[key](params)
        except Exception as e:
            self.logger.error(f"Error processing request {key}: {e}")
            return None
    
    def send_request(self, agent_id: str, method: str, params: Dict[str, Any]) -> Any:
        """
        Alias for request method to maintain compatibility
        
        Args:
            agent_id: Identifier of the target agent
            method: Method to call on the agent
            params: Parameters for the method
        
        Returns:
            Response from the agent or None if request fails
        """
        return self.request(agent_id, method, params)
    
    def unsubscribe(self, topic: str, callback: Optional[Callable] = None):
        """
        Unsubscribe from a topic
        
        Args:
            topic: The topic to unsubscribe from
            callback: Optional specific callback to remove. If None, removes all callbacks for the topic.
        """
        if callback is None:
            # Remove all subscribers for the topic
            self.subscribers[topic].clear()
        else:
            # Remove specific callback
            self.subscribers[topic] = [
                sub for sub in self.subscribers[topic] if sub != callback
            ]
        
        self.logger.debug(f"Unsubscribed from topic: {topic}")
    
    def clear_handlers(self):
        """
        Clear all registered request handlers and subscribers
        """
        self.request_handlers.clear()
        self.subscribers.clear()
        self.logger.debug("Cleared all message bus handlers and subscribers")