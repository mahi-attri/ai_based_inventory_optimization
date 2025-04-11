#!/bin/bash

# Initialize the database if it doesn't exist
if [ ! -f "data/retail_inventory.db" ]; then
    echo "Initializing database..."
    python initialize_db.py
fi

# Import data if needed (uncomment if needed)
# python import_data.py

# Run the main application
echo "Starting the retail inventory optimization system..."
python main.py