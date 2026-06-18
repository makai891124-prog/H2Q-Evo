data = []  # A potentially large list
for i in range(1000): data.append(i)
import pandas as pd
df = pd.read_csv('large_file.csv')