# src/features/outlet_features.py

def add_outlet_features(data):
    # Example: Outlet-specific features
    data['is_large'] = (data['size'] == 'Large').astype(int)
    data['is_high_profile'] = (data['profile'] == 'High').astype(int)
    
    return data
