import matplotlib.pyplot as plt
import pandas as pd
import glob

def plot_csv_data(file_paths, separate_plots=False):
    if separate_plots:
        for file in file_paths:
            data = pd.read_csv(file)
            plt.figure()
            plt.plot(data["Time (s)"], data["Speed (m/s)"], label=file)
            plt.xlabel("Time (s)")
            plt.ylabel("Speed (m/s)")
            plt.title(f"Braking Performance: {file}")
            plt.legend()
            plt.grid()
            plt.show()
    else:
        plt.figure()
        for file in file_paths:
            data = pd.read_csv(file)
            plt.plot(data["Time (s)"], data["Distance (m)"], label=file)
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Braking Performance of Multiple Tests")
        plt.legend()
        plt.grid()
        plt.show()

# Example usage
file_paths = glob.glob("braking_test_*.csv")  # Finds all braking test CSV files
separate_plots = False  # Change to True to generate separate plots
plot_csv_data(file_paths, separate_plots)