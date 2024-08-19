
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
df_min_max_scaled = df_min_max_scaled.add_prefix('minmax_')

# Standard Scaling
standard_scaler = StandardScaler()
df_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
df_standard_scaled = df_standard_scaled.add_prefix('standard_')

# L2 Normalization
normalizer = Normalizer(norm='l2')
df_normalized = pd.DataFrame(normalizer.fit_transform(df), columns=df.columns)
df_normalized = df_normalized.add_prefix('l2norm_')

# Combine all features into one DataFrame
df_combined = pd.concat([df, df_min_max_scaled, df_standard_scaled, df_normalized], axis=1)

# Display the first few rows of the combined DataFrame
print(df_combined.head())

# Save the DataFrame to a CSV file for further use
df_combined.to_csv('scaled_normalized_iris.csv', index=False)
