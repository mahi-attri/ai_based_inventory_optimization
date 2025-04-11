import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional
import requests

class CustomerBehaviorAgent:
    """
    Agent responsible for analyzing customer behavior patterns
    and providing recommendations for marketing and product strategies.
    """
    
    def __init__(self, config: Dict, db_path: str, ollama_base_url: str):
        """
        Initialize the Customer Behavior Agent
        
        Args:
            config: Configuration dictionary for agent settings
            db_path: Path to SQLite database
            ollama_base_url: URL for Ollama API
        """
        self.config = config
        self.db_conn = sqlite3.connect(db_path)
        self.ollama_url = ollama_base_url
        self.message_bus = None
        self.llm_model = config.get("llm_model", "llama3")
        
        # Initialize logging
        self.logger = logging.getLogger("customer_agent")
        
        # Initialize analysis cache
        self.segment_analysis_cache = {}
        self.last_analysis_time = datetime.now() - timedelta(days=7)  # Force initial analysis
    
    def connect_to_message_bus(self, message_bus):
        """Connect to the inter-agent communication bus"""
        self.message_bus = message_bus
        
        try:
            # Subscribe to relevant topics
            self.message_bus.subscribe("pricing_updates", self.handle_pricing_update)
            self.message_bus.subscribe("inventory_updates", self.handle_inventory_update)
            
            # Register request handler
            def safe_customer_handler(params):
                try:
                    return self.handle_customer_request(params)
                except Exception as e:
                    self.logger.error(f"Error in customer request handler: {e}")
                    return {"error": str(e)}
            
            self.message_bus.register_request_handler(
                "customer_agent",
                "get_customer_analysis",
                safe_customer_handler
            )
            
            self.logger.info("Successfully registered customer agent message bus handlers")
        
        except Exception as e:
            self.logger.error(f"Error registering message bus handlers: {e}")
    
    def handle_pricing_update(self, message: Dict):
        """
        Handle pricing update notification
        
        Args:
            message: Message containing pricing update information
        """
        try:
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            old_price = message.get("old_price")
            new_price = message.get("new_price")
            
            if not all([product_id, store_id, old_price, new_price]):
                return
            
            # Calculate price change percentage
            price_change_pct = (new_price - old_price) / old_price
            
            # Log the pricing update
            self.logger.info(f"Received pricing update for product {product_id}, store {store_id}: " 
                           f"{old_price:.2f} → {new_price:.2f} ({price_change_pct:.1%})")
            
            # For significant price changes, predict customer response
            if abs(price_change_pct) > 0.1:  # 10% change
                self.predict_customer_response(product_id, store_id, price_change_pct)
        
        except Exception as e:
            self.logger.error(f"Error in handle_pricing_update: {e}")
    
    def handle_inventory_update(self, message: Dict):
        """
        Handle inventory update notification
        
        Args:
            message: Message containing inventory update information
        """
        try:
            update_type = message.get("type")
            product_id = message.get("product_id")
            store_id = message.get("store_id")
            
            if not product_id or not store_id:
                return
            
            # Only interested in stockout events
            if update_type == "stockout":
                self.logger.info(f"Received stockout notification for product {product_id}, store {store_id}")
                
                # Analyze customer impact of stockout
                self.analyze_stockout_impact(product_id, store_id)
        
        except Exception as e:
            self.logger.error(f"Error in handle_inventory_update: {e}")
    
    def handle_customer_request(self, params: Dict) -> Dict:
        """
        Handle customer analysis request
        
        Args:
            params: Request parameters
            
        Returns:
            Customer analysis data
        """
        try:
            request_type = params.get("type", "segment_analysis")
            
            if request_type == "segment_analysis":
                segment = params.get("segment")
                product_category = params.get("category")
                
                # If segment analysis is cached and recent, return it
                cache_key = f"{segment}_{product_category}"
                if cache_key in self.segment_analysis_cache:
                    cached_result = self.segment_analysis_cache[cache_key]
                    age_hours = (datetime.now() - cached_result["timestamp"]).total_seconds() / 3600
                    
                    if age_hours < 24:  # Cache valid for 24 hours
                        return cached_result
                
                # Perform new segment analysis
                result = self.analyze_customer_segment(segment, product_category)
                
                # Cache the result
                if result and "error" not in result:
                    self.segment_analysis_cache[cache_key] = result
                
                return result
            
            elif request_type == "product_recommendation":
                customer_id = params.get("customer_id")
                
                if not customer_id:
                    return {"error": "Missing customer_id parameter"}
                
                # In a real system, this would use customer purchase history
                # For this demo, we'll generate generic recommendations
                return self.generate_product_recommendations(customer_id)
            
            else:
                return {"error": f"Unknown request type: {request_type}"}
        
        except Exception as e:
            self.logger.error(f"Error in handle_customer_request: {e}")
            return {"error": str(e)}
    
    def predict_customer_response(self, product_id: int, store_id: int, price_change_pct: float):
        """
        Predict customer response to a price change
        
        Args:
            product_id: Product identifier
            store_id: Store identifier
            price_change_pct: Percentage price change
        """
        try:
            # Get product information
            cursor = self.db_conn.cursor()
            cursor.execute("""
            SELECT p.Name, p.Category, po.Price, po.Elasticity_Index
            FROM products p
            JOIN pricing_optimization po ON p.Product_ID = po.Product_ID
            WHERE p.Product_ID = ? AND po.Store_ID = ?
            """, (product_id, store_id))
            
            result = cursor.fetchone()
            
            if not result:
                return
            
            name, category, price, elasticity = result
            
            # Calculate expected demand change
            expected_demand_change = -1 * elasticity * price_change_pct
            
            # Get recent sales data
            cursor.execute("""
            SELECT AVG(Sales_Quantity) as avg_sales
            FROM demand_forecasting
            WHERE Product_ID = ? AND Store_ID = ?
            AND Date >= date('now', '-30 days')
            """, (product_id, store_id))
            
            avg_sales_result = cursor.fetchone()
            avg_sales = avg_sales_result[0] if avg_sales_result and avg_sales_result[0] else 0
            
            # Calculate expected sales change
            expected_sales_change = avg_sales * expected_demand_change
            
            self.logger.info(f"Predicted customer response for {name} price change ({price_change_pct:.1%}):")
            self.logger.info(f"- Expected demand change: {expected_demand_change:.1%}")
            self.logger.info(f"- Average daily sales: {avg_sales:.1f} units")
            self.logger.info(f"- Expected sales change: {expected_sales_change:.1f} units per day")
            
            # Publish customer response prediction
            if self.message_bus:
                self.message_bus.publish(
                    "customer_insights",
                    {
                        "type": "price_response_prediction",
                        "product_id": product_id,
                        "store_id": store_id,
                        "product_name": name,
                        "category": category,
                        "price_change_pct": price_change_pct,
                        "expected_demand_change_pct": expected_demand_change,
                        "expected_sales_change": expected_sales_change,
                        "confidence": 70,  # Hardcoded confidence for demo
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Error in predict_customer_response: {e}")
    
    def analyze_stockout_impact(self, product_id: int, store_id: int):
        """
        Analyze customer impact of a product stockout
        
        Args:
            product_id: Product identifier
            store_id: Store identifier
        """
        try:
            # Get product information
            cursor = self.db_conn.cursor()
            cursor.execute("""
            SELECT p.Name, p.Category, p.Subcategory
            FROM products p
            WHERE p.Product_ID = ?
            """, (product_id,))
            
            product_result = cursor.fetchone()
            
            if not product_result:
                return
            
            name, category, subcategory = product_result
            
            # Get alternative products (same category and subcategory)
            cursor.execute("""
            SELECT p.Product_ID, p.Name, po.Price
            FROM products p
            JOIN pricing_optimization po ON p.Product_ID = po.Product_ID
            WHERE p.Category = ? AND p.Subcategory = ?
            AND p.Product_ID != ? AND po.Store_ID = ?
            LIMIT 3
            """, (category, subcategory, product_id, store_id))
            
            alternatives = cursor.fetchall()
            
            # Prepare alternative product data
            alt_products = [
                {
                    "product_id": alt[0],
                    "name": alt[1],
                    "price": alt[2]
                }
                for alt in alternatives
            ]
            
            # Calculate loyalty impact score (0-100)
            # In a real system, this would be based on customer purchase history
            loyalty_impact = min(80, max(20, 50 + len(alt_products) * -10))
            
            self.logger.info(f"Analyzed stockout impact for {name}:")
            self.logger.info(f"- Alternative products: {len(alt_products)}")
            self.logger.info(f"- Customer loyalty impact score: {loyalty_impact}")
            
            # Publish stockout impact analysis
            if self.message_bus:
                self.message_bus.publish(
                    "customer_insights",
                    {
                        "type": "stockout_impact",
                        "product_id": product_id,
                        "store_id": store_id,
                        "product_name": name,
                        "category": category,
                        "subcategory": subcategory,
                        "alternative_products": alt_products,
                        "loyalty_impact_score": loyalty_impact,
                        "recommendations": [
                            "Offer discount on alternative products",
                            "Prioritize restocking this item"
                        ] if loyalty_impact > 50 else [
                            "Promote alternative products",
                            "Normal restocking priority"
                        ],
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Error in analyze_stockout_impact: {e}")
    
    def analyze_customer_segment(self, segment: str = None, product_category: str = None) -> Dict:
        """
        Analyze customer segment behavior
        
        Args:
            segment: Customer segment to analyze
            product_category: Product category to focus on
            
        Returns:
            Segment analysis data
        """
        try:
            # Default to all segments and categories if not specified
            if not segment:
                segment = "all"
            
            # Get product category sales data
            category_clause = ""
            params = []
            
            if product_category:
                category_clause = """
                JOIN products p ON df.Product_ID = p.Product_ID
                WHERE p.Category = ?
                """
                params.append(product_category)
            
            # Query to get sales data with customer segments
            query = f"""
            SELECT df.Date, df.Sales_Quantity, df.Price, df.Customer_Segments
            FROM demand_forecasting df
            {category_clause}
            ORDER BY df.Date DESC
            LIMIT 1000
            """
            
            df = pd.read_sql_query(query, self.db_conn, params=params)
            
            if df.empty:
                return {
                    "error": "No data available for the specified segment and category"
                }
            
            # Parse customer segments JSON
            segment_data = []
            for idx, row in df.iterrows():
                try:
                    segments = json.loads(row['Customer_Segments'])
                    segment_data.append({
                        'date': row['Date'],
                        'sales': row['Sales_Quantity'],
                        'price': row['Price'],
                        'segments': segments
                    })
                except:
                    pass
            
            if not segment_data:
                return {
                    "error": "No valid segment data found"
                }
            
            # Calculate segment metrics
            if segment == "all":
                # Calculate overall metrics
                total_sales = sum(item['sales'] for item in segment_data)
                avg_price = np.mean([item['price'] for item in segment_data])
                
                # Calculate segment distribution
                segment_names = list(segment_data[0]['segments'].keys())
                segment_distributions = {}
                
                for seg_name in segment_names:
                    values = [item['segments'].get(seg_name, 0) for item in segment_data]
                    segment_distributions[seg_name] = np.mean(values)
                
                # Generate insights
                insights = [
                    f"Average price point: ${avg_price:.2f}",
                    f"Segment distribution: {', '.join([f'{k}: {v:.1%}' for k, v in segment_distributions.items()])}"
                ]
                
                # Add category-specific insight if provided
                if product_category:
                    insights.append(f"{product_category} category sees strong engagement across all segments")
                
                recommendations = self._generate_segment_recommendations(segment, product_category, segment_distributions)
                
                return {
                    "segment": "all",
                    "product_category": product_category if product_category else "all",
                    "total_sales": total_sales,
                    "average_price": avg_price,
                    "segment_distribution": segment_distributions,
                    "insights": insights,
                    "recommendations": recommendations,
                    "timestamp": datetime.now()
                }
            else:
                # Filter for specific segment
                filtered_data = []
                for item in segment_data:
                    if segment in item['segments']:
                        # Add segment percentage to the item
                        item['segment_pct'] = item['segments'][segment]
                        filtered_data.append(item)
                
                if not filtered_data:
                    return {
                        "error": f"No data for segment '{segment}'"
                    }
                
                # Calculate segment-specific metrics
                total_sales = sum(item['sales'] * item['segment_pct'] for item in filtered_data)
                avg_price = np.mean([item['price'] for item in filtered_data])
                
                # Calculate price sensitivity
                price_sensitivity = self._calculate_price_sensitivity(filtered_data, segment)
                
                # Generate insights
                insights = [
                    f"Average price point for {segment} segment: ${avg_price:.2f}",
                    f"Price sensitivity: {price_sensitivity:.2f} (higher means more sensitive)",
                ]
                
                # Add category-specific insight if provided
                if product_category:
                    insights.append(f"{segment} segment shows {'high' if price_sensitivity > 1.2 else 'moderate' if price_sensitivity > 0.8 else 'low'} price sensitivity for {product_category}")
                
                # Generate recommendations
                segment_distribution = {segment: 1.0}
                recommendations = self._generate_segment_recommendations(segment, product_category, segment_distribution, price_sensitivity)
                
                return {
                    "segment": segment,
                    "product_category": product_category if product_category else "all",
                    "total_attributed_sales": total_sales,
                    "average_price": avg_price,
                    "price_sensitivity": price_sensitivity,
                    "insights": insights,
                    "recommendations": recommendations,
                    "timestamp": datetime.now()
                }
        
        except Exception as e:
            self.logger.error(f"Error in analyze_customer_segment: {e}")
            return {"error": str(e)}
    
    def _calculate_price_sensitivity(self, data: List[Dict], segment: str) -> float:
        """
        Calculate price sensitivity for a segment
        
        Args:
            data: List of data points with sales, price, and segment percentage
            segment: Segment name
            
        Returns:
            Price sensitivity score (higher means more sensitive)
        """
        if len(data) < 10:
            return 1.0  # Default value for insufficient data
        
        # Group by price ranges
        price_ranges = {}
        price_min = min(item['price'] for item in data)
        price_max = max(item['price'] for item in data)
        
        range_size = (price_max - price_min) / 5  # 5 price ranges
        
        for item in data:
            price = item['price']
            range_idx = min(4, int((price - price_min) / range_size))
            
            if range_idx not in price_ranges:
                price_ranges[range_idx] = {
                    'min_price': price_min + range_idx * range_size,
                    'max_price': price_min + (range_idx + 1) * range_size,
                    'sales': [],
                    'segment_pct': []
                }
            
            price_ranges[range_idx]['sales'].append(item['sales'])
            price_ranges[range_idx]['segment_pct'].append(item['segment_pct'])
        
        # Calculate average sales * segment_pct for each range
        range_metrics = []
        for range_idx, range_data in price_ranges.items():
            if not range_data['sales']:
                continue
                
            avg_price = (range_data['min_price'] + range_data['max_price']) / 2
            avg_sales = np.mean([s * p for s, p in zip(range_data['sales'], range_data['segment_pct'])])
            
            range_metrics.append({
                'avg_price': avg_price,
                'avg_sales': avg_sales
            })
        
        if len(range_metrics) < 2:
            return 1.0  # Default value for insufficient price ranges
        
        # Sort by price
        range_metrics.sort(key=lambda x: x['avg_price'])
        
        # Calculate slope
        prices = [r['avg_price'] for r in range_metrics]
        sales = [r['avg_sales'] for r in range_metrics]
        
        # Normalize
        prices_norm = np.array(prices) / np.mean(prices)
        sales_norm = np.array(sales) / np.mean(sales)
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(prices_norm, sales_norm)[0, 1]
            
            # Convert to sensitivity score (higher means more sensitive to price)
            sensitivity = max(0.5, min(2.0, abs(correlation) * 2))
            
            # If correlation is positive (higher prices -> higher sales), reduce sensitivity
            if correlation > 0:
                sensitivity *= 0.5
            
            return sensitivity
            
        except:
            return 1.0  # Default value on error
    
    def _generate_segment_recommendations(self, segment: str, product_category: str, 
                                        segment_distribution: Dict, price_sensitivity: float = None) -> List[str]:
        """
        Generate recommendations for a customer segment
        
        Args:
            segment: Customer segment
            product_category: Product category
            segment_distribution: Distribution of segments
            price_sensitivity: Price sensitivity score (optional)
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if segment == "all":
            # Recommendations based on segment distribution
            most_popular = max(segment_distribution.items(), key=lambda x: x[1])
            least_popular = min(segment_distribution.items(), key=lambda x: x[1])
            
            recommendations.append(f"Focus marketing efforts on the {most_popular[0]} segment")
            recommendations.append(f"Develop strategies to increase engagement with the {least_popular[0]} segment")
            
            if product_category:
                recommendations.append(f"Consider targeted promotions for {product_category} products to specific segments")
        else:
            # Segment-specific recommendations
            if price_sensitivity and price_sensitivity > 1.2:
                recommendations.append(f"The {segment} segment is highly price sensitive")
                recommendations.append(f"Consider targeted discounts or loyalty programs for {segment} customers")
            elif price_sensitivity and price_sensitivity < 0.8:
                recommendations.append(f"The {segment} segment shows low price sensitivity")
                recommendations.append(f"Focus on quality and features rather than discounts for {segment} customers")
            else:
                recommendations.append(f"The {segment} segment shows moderate price sensitivity")
                recommendations.append(f"Balance price and quality factors for the {segment} segment")
            
            if product_category:
                recommendations.append(f"Customize {product_category} product displays and messaging for {segment} segment preferences")
        
        # Add LLM-enhanced recommendations if available
        enhanced_recommendations = self._enhance_recommendations_with_llm(segment, product_category, price_sensitivity)
        if enhanced_recommendations:
            recommendations.extend(enhanced_recommendations)
        
        return recommendations
    
    def _enhance_recommendations_with_llm(self, segment: str, product_category: str, price_sensitivity: float = None) -> List[str]:
        """
        Enhance recommendations using LLM
        
        Args:
            segment: Customer segment
            product_category: Product category
            price_sensitivity: Price sensitivity score (optional)
            
        Returns:
            List of enhanced recommendations
        """
        try:
            # Skip if not configured to use LLM
            use_llm = self.config.get("use_llm_for_recommendations", False)
            if not use_llm:
                return []
            
            # Create prompt
            prompt = f"""
            You are a retail customer behavior expert. Please provide 2-3 specific marketing and merchandising 
            recommendations for the following customer segment and product category:
            
            Customer Segment: {segment}
            Product Category: {product_category or 'All categories'}
            """
            
            if price_sensitivity is not None:
                prompt += f"""
                Price Sensitivity: {'High' if price_sensitivity > 1.2 else 'Moderate' if price_sensitivity > 0.8 else 'Low'}
                """
            
            prompt += """
            Focus on actionable, specific suggestions that retailers could implement.
            Provide your recommendations as a bullet-point list, with each point starting with a dash (-).
            """
            
            # Call LLM
            response = self._call_ollama_api(prompt)
            
            # Parse recommendations
            recommendations = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    recommendations.append(line[2:])
            
            return recommendations[:3]  # Limit to 3 recommendations
            
        except Exception as e:
            self.logger.error(f"Error in _enhance_recommendations_with_llm: {e}")
            return []
    
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
                    "temperature": 0.2,  # Low temperature for more consistent responses
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
    
    def generate_product_recommendations(self, customer_id: int) -> Dict:
        """
        Generate product recommendations for a customer
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Product recommendation data
        """
        try:
            # In a real system, this would be based on customer purchase history
            # For this demo, we'll generate random recommendations from top-selling products
            
            # Get top-selling products
            cursor = self.db_conn.cursor()
            cursor.execute("""
            SELECT p.Product_ID, p.Name, p.Category, p.Subcategory, AVG(df.Sales_Quantity) as avg_sales
            FROM products p
            JOIN demand_forecasting df ON p.Product_ID = df.Product_ID
            GROUP BY p.Product_ID
            ORDER BY avg_sales DESC
            LIMIT 20
            """)
            
            top_products = cursor.fetchall()
            
            if not top_products:
                return {
                    "customer_id": customer_id,
                    "recommendations": [],
                    "error": "No product data available"
                }
            
            # Randomly select 5 products
            np.random.seed(customer_id)  # Seed for reproducibility
            selected_indices = np.random.choice(len(top_products), min(5, len(top_products)), replace=False)
            
            recommendations = []
            for idx in selected_indices:
                product = top_products[idx]
                
                recommendations.append({
                    "product_id": product[0],
                    "name": product[1],
                    "category": product[2],
                    "subcategory": product[3],
                    "relevance_score": np.random.uniform(0.5, 0.95)  # Random relevance score
                })
            
            # Sort by relevance
            recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "customer_id": customer_id,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in generate_product_recommendations: {e}")
            return {
                "customer_id": customer_id,
                "recommendations": [],
                "error": str(e)
            }
    
    def analyze_purchase_patterns(self):
        """
        Analyze customer purchase patterns
        
        Returns:
            Dict with analysis results
        """
        try:
            # In a real system, this would analyze customer purchase data
            # For this demo, we'll simulate an analysis based on sales data
            
            # Get sales by day of week
            cursor = self.db_conn.cursor()
            cursor.execute("""
            SELECT strftime('%w', Date) as day_of_week, 
                   SUM(Sales_Quantity) as total_sales
            FROM demand_forecasting
            WHERE Date >= date('now', '-90 days')
            GROUP BY day_of_week
            ORDER BY day_of_week
            """)
            
            day_of_week_sales = cursor.fetchall()
            
            # Get sales by product category
            cursor.execute("""
            SELECT p.Category, SUM(df.Sales_Quantity) as total_sales
            FROM demand_forecasting df
            JOIN products p ON df.Product_ID = p.Product_ID
            WHERE df.Date >= date('now', '-90 days')
            GROUP BY p.Category
            ORDER BY total_sales DESC
            """)
            
            category_sales = cursor.fetchall()
            
            # Format results
            days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            day_sales = [{"day": days[int(row[0])], "sales": row[1]} for row in day_of_week_sales]
            
            cat_sales = [{"category": row[0], "sales": row[1]} for row in category_sales]
            
            # Identify peak sales day
            peak_day = max(day_sales, key=lambda x: x["sales"])
            
            # Identify top category
            top_category = cat_sales[0] if cat_sales else {"category": "unknown", "sales": 0}
            
            # Generate insights
            insights = [
                f"Peak sales day is {peak_day['day']}",
                f"Top-selling category is {top_category['category']}"
            ]
            
            # Generate recommendations based on patterns
            recommendations = []
            
            # Recommend staffing based on peak day
            recommendations.append(f"Increase staffing on {peak_day['day']} to handle higher sales volume")
            
            # Recommend featuring top category
            recommendations.append(f"Feature {top_category['category']} products in prominent store locations")
            
            # Recommend for low sales days
            min_day = min(day_sales, key=lambda x: x["sales"])
            recommendations.append(f"Run promotions on {min_day['day']} to increase store traffic")
            
            return {
                "day_of_week_analysis": day_sales,
                "category_analysis": cat_sales,
                "insights": insights,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_purchase_patterns: {e}")
            return {"error": str(e)}
    
    def run_periodic_customer_analysis(self):
        """
        Run periodic customer analysis
        
        Returns:
            Dict with analysis results
        """
        try:
            self.logger.info("Running periodic customer analysis")
            
            # Record start time
            start_time = datetime.now()
            
            # Run segment analysis for each major segment
            segment_analyses = {}
            for segment in ["regular", "premium", "budget"]:
                try:
                    result = self.analyze_customer_segment(segment)
                    if "error" not in result:
                        segment_analyses[segment] = result
                except Exception as e:
                    self.logger.error(f"Error analyzing {segment} segment: {e}")
            
            # Run purchase pattern analysis
            purchase_patterns = self.analyze_purchase_patterns()
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Update last analysis time
            self.last_analysis_time = datetime.now()
            
            # Publish key insights
            if self.message_bus:
                insights = []
                
                # Add segment insights
                for segment, analysis in segment_analyses.items():
                    if "insights" in analysis:
                        for insight in analysis["insights"][:1]:  # Just the first insight
                            insights.append(f"{segment}: {insight}")
                
                # Add pattern insights
                if "insights" in purchase_patterns and not isinstance(purchase_patterns["insights"], str):
                    insights.extend(purchase_patterns["insights"])
                
                # Publish insights
                if insights:
                    self.message_bus.publish(
                        "customer_insights",
                        {
                            "type": "periodic_analysis",
                            "insights": insights,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            
            return {
                "segment_analyses": segment_analyses,
                "purchase_patterns": purchase_patterns,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Error in run_periodic_customer_analysis: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """
        Perform cleanup operations
        """
        try:
            if self.db_conn:
                self.db_conn.close()
            
            self.logger.info("Customer Agent cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")