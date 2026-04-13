import pandas as pd

def preprocess_data(df):
    print("✅ NEW preprocess file is running")  # debug line

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # FIXED LINE
    df = df.ffill()

    return df