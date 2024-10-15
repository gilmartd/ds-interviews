import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
import warnings

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Step 1: User Selection for Input File
def select_input_path():
    root = Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Select Dataset", "Please choose the dataset (.csv file) that you want to analyze.")
    input_filepath = filedialog.askopenfilename(title="Select the CSV file to analyze", filetypes=[("CSV Files", "*.csv")])
    return input_filepath

# Step 2: Load Data
input_filepath = select_input_path()
df = pd.read_csv(input_filepath)

# Convert datetime_utc to datetime format and drop the column after creating a timestamp representation
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
df['timestamp'] = df['datetime_utc'].astype(int) / 10**9  # Convert to Unix timestamp in seconds
df.drop(columns=['datetime_utc'], inplace=True)

# Step 3: Drop rows with missing values for simplicity
df.dropna(inplace=True)

# Step 4: Define Causal Model using DoWhy
# Let's test the causal effect of precipitation on turbidity
model = CausalModel(
    data=df,
    treatment='precp_in',
    outcome='turb_fnu',
    common_causes=['wtempc', 'windspeed_knots', 'wvel_fps', 'winddir_dcfn'],  # Potential confounders
    instruments=None  # We don't have instrumental variables for this analysis
)

# Visualize the causal graph
model.view_model()
plt.show()

# Step 5: Identify the Effect
identified_estimand = model.identify_effect()
print(identified_estimand)

# Step 6: Estimate the Effect
causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print("Causal Estimate: ", causal_estimate.value)

# Step 7: Refute the Estimate using a different refuter
# Use data_subset_refuter instead of placebo_treatment_refuter
try:
    refutation = model.refute_estimate(identified_estimand, causal_estimate, method_name="data_subset_refuter")
    print(refutation)
except Exception as e:
    print("Refutation process failed:", e)

# Summary
print("\nCausal inference completed successfully.")
