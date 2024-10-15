import pandas as pd
import matplotlib.pyplot as plt
from tkinter import simpledialog, Tk

# Load the dataset
df = pd.read_csv('dev.csv')

# Dictionary mapping column names to descriptive labels
column_descriptions = {
    'datetime_utc': 'Timestamp for observations given in UTC',
    'wtempc': 'Water temperature in degrees Celsius',
    'atempc': 'Air temperature at water surface in degrees Celsius',
    'winddir_dcfn': 'Wind direction in degrees clockwise from North',
    'precp_in': 'Precipitation in inches',
    'relh_pct': 'Relative humidity (percent of saturation)',
    'spc': 'Specific conductivity (microsiemens/centimeter)',
    'dox_mgl': 'Dissolved oxygen in milligrams per liter',
    'ph': 'pH of water in standard units (SU, feasible range of 0-14)',
    'windgust_knots': 'Speed of wind gusts in knots',
    'wse1988': 'Water surface elevation in NAVD88 datum',
    'wvel_fps': 'Water velocity in feet per second',
    'mbars': 'Atmospheric pressure in millibars',
    'windspeed_knots': 'Average wind speed in knots',
    'par': 'Photosynthetically available radiation (millimoles of photons per square meter)',
    'turb_fnu': 'Water turbidity (formazin nephelometric units)'
}

# Step 5: User-Selected Feature Exploration Loop
def compare_features():
    root = Tk()
    root.withdraw()  # Hide the root window

    while True:
        # Display available columns for user reference
        print("\nAvailable Columns for Comparison:")
        for key, value in column_descriptions.items():
            print(f"{key}: {value}")

        # Ask the user to input the names of two columns to compare
        col1 = simpledialog.askstring("Input", "Enter the name of the first column to compare:")
        col2 = simpledialog.askstring("Input", "Enter the name of the second column to compare:")

        # Validate user input
        if col1 not in column_descriptions or col2 not in column_descriptions:
            print("Invalid column names entered. Please try again.")
            continue

        # Create scatter plot for the selected columns
        print(f"\nExploring relationship between '{column_descriptions[col1]}' and '{column_descriptions[col2]}'...")
        plt.figure(figsize=(10, 6))
        plt.scatter(df[col1], df[col2], alpha=0.5)
        plt.xlabel(column_descriptions[col1])
        plt.ylabel(column_descriptions[col2])
        plt.title(f'{column_descriptions[col1]} vs. {column_descriptions[col2]}')
        plt.tight_layout()
        plt.show()

        # Ask the user if they want to compare more columns
        more_comparisons = simpledialog.askstring("Input", "Do you want to compare more columns? (yes/no):")
        if more_comparisons.lower() != 'yes':
            print("Moving on to the next step...")
            break

# Run the feature comparison loop
compare_features()
