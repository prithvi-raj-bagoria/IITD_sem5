import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np

def shorten_test_case_name(name):
    """Convert test case names to shortened format (e.g., 'easy_1.txt' to 'e_1.txt')"""
    if name.startswith('easy_'):
        return 'e' + name[4:]
    elif name.startswith('medium_'):
        return 'm' + name[6:]
    elif name.startswith('hard_'):
        return 'h' + name[4:]
    return name

def main():
    # Read the CSV data
    try:
        df = pd.read_csv('plot_data.csv')
    except FileNotFoundError:
        print("Error: plot_data.csv not found. Run the test program first.")
        return
    
    # Set up a larger figure size for better visibility
    plt.figure(figsize=(10, 7))
    
    # Configure the plot style with fewer gridlines
    sns.set_style("whitegrid")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Sort by test case name
    df = df.sort_values('TestCase')
    
    # Create a line plot with enhanced visibility
    ax = sns.lineplot(
        x=range(len(df)), 
        y='ObjectiveValue', 
        data=df,
        marker='o',
        markersize=10,
        color='royalblue',
        linewidth=2.5
    )
    
    # Customize the plot
    plt.title('Objective Value by Test Case', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Test Case', fontsize=16, labelpad=10)
    plt.ylabel('Objective Value', fontsize=16, labelpad=10)
    
    # Set x-tick positions and labels with better spacing
    tick_positions = range(len(df))
    # Apply shortening to test case names
    tick_labels = [shorten_test_case_name(name) for name in df['TestCase'].tolist()]
    
    # If there are too many test cases, show only every nth label
    n = max(1, len(tick_labels) // 15)  # Show at most ~15 labels
    plt.xticks(
        tick_positions[::n], 
        tick_labels[::n], 
        rotation=45, 
        ha='right', 
        fontsize=10
    )
    
    # Add value labels with improved positioning
    for i, v in enumerate(df['ObjectiveValue']):
        # Only label every nth point if there are many
        if i % n == 0:
            ax.text(
                i, v + (df['ObjectiveValue'].max() * 0.02),  # Position slightly above the point
                f"{v:.1f}", 
                ha='center', 
                fontsize=9,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    # Add average line with better visibility
    avg_value = df['ObjectiveValue'].mean()
    plt.axhline(y=avg_value, color='red', linestyle='--', linewidth=2,
                label=f'Average: {avg_value:.1f}')
    
    # Add min/max reference lines
    min_value = df['ObjectiveValue'].min()
    max_value = df['ObjectiveValue'].max()
    plt.axhline(y=min_value, color='orange', linestyle=':', linewidth=1.5,
                label=f'Min: {min_value:.1f}')
    plt.axhline(y=max_value, color='green', linestyle=':', linewidth=1.5,
                label=f'Max: {max_value:.1f}')
    
    # Add legend with better positioning
    plt.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)
    
    # Set y-axis limits with some padding
    y_min = max(0, df['ObjectiveValue'].min() * 0.9)  # Don't go below zero
    y_max = df['ObjectiveValue'].max() * 1.1
    plt.ylim(y_min, y_max)
    
    # Adjust layout to fit everything
    plt.tight_layout()
    
    # Save the plot with high DPI for better quality
    plt.savefig('objective_values.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'objective_values.png'")
    
    # Show the plot if not in non-interactive mode
    plt.show()

if __name__ == "__main__":
    main()
