# src/features/item_features.py

def add_item_features(data):
    # Example: Average sales per item
    data['avg_sales_per_item'] = data['net_sales'] / data['item_qty']
    
    return data
