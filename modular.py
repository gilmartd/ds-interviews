import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from tkinter import Tk, filedialog, messagebox

# Step 1: User Selection for Input File and Output Directory
def select_input_output_paths():
    """Open dialog for user to select input CSV and output directory with clear prompts."""
    root = Tk()
    root.withdraw()  # Hide the root window

    # Display a message to choose the dataset file
    messagebox.showinfo("Select Dataset", "Please choose the dataset (.csv file) that you want to analyze.")
    
    # Ask user to select the CSV file
    input_filepath = filedialog.askopenfilename(
        title="Select the CSV file to analyze",
        filetypes=[("CSV Files", "*.csv")]
    )

    # Display a message to choose the output folder
    messagebox.showinfo("Select Output Folder", "Please choose the folder where you want to save the output results.")

    # Ask user to select the output directory
    output_dir = filedialog.askdirectory(
        title="Select the output directory for saving results"
    )

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return input_filepath, output_dir

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

# Step 2: Load Data
def load_data(filepath):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(filepath)

# Step 3: Cap Outliers
def cap_outliers(df, lower_quantile=0.05, upper_quantile=0.95):
    """Cap outliers in the numerical columns of the DataFrame at the specified quantiles."""
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_columns:
        lower_limit = df[col].quantile(lower_quantile)
        upper_limit = df[col].quantile(upper_quantile)
        df[col] = np.clip(df[col], lower_limit, upper_limit)
    
    return df

# Step 4: Generate Summary Statistics
def generate_summary_stats(df, output_dir):
    """Generate summary statistics for numerical features in the DataFrame and save to CSV."""
    summary_stats = df.describe().transpose()
    numeric_columns = df.select_dtypes(include=['number'])
    summary_stats['skewness'] = numeric_columns.skew()
    summary_stats['unique_values'] = df.nunique()

    output_path = os.path.join(output_dir, 'summary_statistics.csv')
    print(f"Saving summary statistics to: {output_path}")

    try:
        # Save the summary statistics to CSV
        summary_stats.to_csv(output_path)
    except PermissionError:
        print(f"Permission denied: Unable to write to {output_path}. Please make sure the file is not open in another program.")

    return summary_stats

# Step 5: Summarize Completeness
def summarize_completeness(df, output_dir):
    """Summarize the completeness (missing values) of each feature in the DataFrame and save to CSV."""
    completeness_summary = df.isnull().sum() / len(df) * 100
    completeness_table = pd.DataFrame({
        'missing_values_percentage': completeness_summary,
        'non_null_count': df.notnull().sum(),
        'total_count': len(df)
    })

    output_path = os.path.join(output_dir, 'completeness_summary.csv')
    print(f"Saving completeness summary to: {output_path}")

    try:
        # Save the completeness table to CSV
        completeness_table.to_csv(output_path)
    except PermissionError:
        print(f"Permission denied: Unable to write to {output_path}. Please make sure the file is not open in another program.")

    return completeness_table

# Step 6: Visualize Completeness and Save
def visualize_completeness(completeness_summary, output_dir):
    """Create a bar chart for missing values percentage of each feature and save the plot as PNG."""
    # Define colors based on the condition (> 1% missing values)
    colors = ['red' if value > 1 else 'skyblue' for value in completeness_summary]

    plt.figure(figsize=(12, 6))
    completeness_summary.plot(kind='bar', color=colors)
    plt.title('Percentage of Missing Values per Feature')
    plt.xlabel('Feature')
    plt.ylabel('Percentage of Missing Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'missing_values_percentage.png')
    print(f"Saving completeness chart to: {output_path}")

    plt.savefig(output_path)
    plt.close()

# Step 7: Visualize Distributions and Save
def visualize_distributions(df, output_dir, charts_per_page=6):
    """Create histograms for each numeric feature, distributed across multiple pages, and save plots as PNGs."""
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    num_chunks = -(-len(numeric_columns) // charts_per_page)

    for i in range(num_chunks):
        start = i * charts_per_page
        end = start + charts_per_page
        current_chunk = numeric_columns[start:end]

        num_cols = 3  # Number of columns for subplots on each page
        num_rows = -(-len(current_chunk) // num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        # Plot histograms for the current chunk
        for j, col in enumerate(current_chunk):
            df[col].hist(bins=30, color='dodgerblue', alpha=0.7, ax=axes[j])
            descriptive_title = column_descriptions.get(col, col)  # Use descriptive title if available, otherwise default to column name
            axes[j].set_title(f'{descriptive_title}')
            axes[j].set_xlabel(descriptive_title)
            axes[j].set_ylabel('Frequency')

        # Hide any empty subplots
        for k in range(len(current_chunk), len(axes)):
            fig.delaxes(axes[k])

        plt.tight_layout()

        output_path = os.path.join(output_dir, f'distributions_page_{i + 1}.png')
        print(f"Saving distribution chart to: {output_path}")

        # Save the plot as a PNG file
        plt.savefig(output_path)
        plt.close()

# Main Script
if __name__ == "__main__":
    # Ask user to select input CSV file and output directory
    input_filepath, output_dir = select_input_output_paths()

    # Load data
    df = load_data(input_filepath)

    # Cap outliers
    df = cap_outliers(df, lower_quantile=0.05, upper_quantile=0.95)

    # Generate summary statistics
    summary_stats = generate_summary_stats(df, output_dir)
    print(summary_stats)

    # Summarize completeness
    completeness_table = summarize_completeness(df, output_dir)
    print(completeness_table)

    # Visualize completeness and save the plot
    visualize_completeness(completeness_table['missing_values_percentage'], output_dir)

    # Visualize feature distributions across multiple pages and save the plots
    visualize_distributions(df, output_dir, charts_per_page=6)
