import pandas as pd

def analyze_data(data):
    df = pd.DataFrame(data)
    # Perform some basic analysis
    description = df.describe()
    return description.to_string()