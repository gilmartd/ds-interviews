import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tkinter import Tk, filedialog

# Use file dialog to ask user for the dataset file path
Tk().withdraw()  # Hide the root window
data_path = filedialog.askopenfilename(title="Select the dataset CSV file", filetypes=[("CSV files", "*.csv")])
df = pd.read_csv(data_path)

# Use file dialog to ask user for the folder to save the graphs
graph_folder = filedialog.askdirectory(title="Select the folder to save the graphs")
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

# Convert datetime_utc to datetime format and set it as the index
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
df.set_index('datetime_utc', inplace=True)

# Step 1: Seasonal Patterns and Trends - Temperature Analysis
print("Step 1: Analyzing seasonal patterns for water and air temperature...")
monthly_temp = df[['wtempc', 'atempc']].resample('M').mean()
plt.figure(figsize=(10, 6))
monthly_temp.plot(title='Monthly Average Water and Air Temperature', marker='o')
plt.ylabel('Temperature (°C)')
plt.xlabel('Month')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(graph_folder, 'monthly_avg_temperature.png'))
plt.show()

# Step 2: Correlation Analysis - Correlation Matrix
print("Step 2: Analyzing correlations between features...")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix for Numerical Features')
plt.tight_layout()
plt.savefig(os.path.join(graph_folder, 'correlation_matrix.png'))
plt.show()

# Step 3: Missing Data Patterns - Missing Data Over Time
print("Step 3: Analyzing missing data patterns over time...")
missing_over_time = df.isnull().resample('M').sum()
plt.figure(figsize=(12, 6))
missing_over_time.plot(kind='bar', stacked=True, title='Missing Data per Feature Over Time')
plt.xlabel('Month')
plt.ylabel('Count of Missing Values')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(graph_folder, 'missing_data_over_time.png'))
plt.show()

# Step 4: Explore Potential Outliers - Boxplots
print("Step 4: Identifying potential outliers for selected features...")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['precp_in', 'windspeed_knots', 'wtempc']])
plt.title('Boxplot for Selected Features to Identify Outliers')
plt.ylabel('Values')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(graph_folder, 'outliers_boxplot.png'))
plt.show()

# Step 5: Feature-Specific Exploration - Dissolved Oxygen vs. Water Temperature
print("Step 5: Exploring relationship between water temperature and dissolved oxygen...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wtempc', y='dox_mgl', data=df, alpha=0.5)
plt.xlabel('Water Temperature (°C)')
plt.ylabel('Dissolved Oxygen (mg/L)')
plt.title('Dissolved Oxygen vs. Water Temperature')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(graph_folder, 'dox_vs_wtempc.png'))
plt.show()

# Step 6: Seasonal Analysis of pH, PAR, and Dissolved Oxygen
print("Step 6: Analyzing seasonal characteristics for pH, PAR, and dissolved oxygen...")
monthly_avg = df[['pH', 'PAR', 'dox_mgl']].resample('M').mean()
plt.figure(figsize=(12, 6))
monthly_avg.plot(title='Monthly Average of pH, PAR, and Dissolved Oxygen', marker='o')
plt.ylabel('Values')
plt.xlabel('Month')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(graph_folder, 'monthly_avg_pH_PAR_dox.png'))
plt.show()

# Final: Summary of Steps Completed
print("\nAll steps completed successfully:")
print("1. Seasonal trends of temperature analyzed.")
print("2. Correlation analysis completed.")
print("3. Missing data patterns over time visualized.")
print("4. Boxplots for outlier identification created.")
print("5. Relationship between water temperature and dissolved oxygen analyzed.")
print("6. Seasonal characteristics of pH, PAR, and dissolved oxygen analyzed.")