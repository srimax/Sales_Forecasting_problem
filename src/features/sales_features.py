
def add_sales_features(data):
    # Example: Cumulative sales by store and department
    data['cum_sales'] = data.groupby(['store', 'item_dept'])['net_sales'].cumsum()
    
    # Example: Rolling mean of item_qty over the last 7 days
    rolling_mean_qty = (
        data.groupby(['store', 'item_dept'])['item_qty']
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)  # Align with original index
    )
    
    # Assign the result back to the original DataFrame
    data['rolling_mean_qty'] = rolling_mean_qty
    
    return data