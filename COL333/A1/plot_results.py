import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def main():
    # Read the CSV data
    try:
        df = pd.read_csv('plot_data.csv')
    except FileNotFoundError:
        print("Error: plot_data.csv not found. Run the test program first.")
        return
    
    # Set up the figure size
    plt.figure(figsize=(12, 8))
    
    # Configure the plot style
    sns.set_style("whitegrid")
    
    # Sort by test case name
    df = df.sort_values('TestCase')
    
    # Plot the objective values
    ax = sns.barplot(x='TestCase', y='ObjectiveValue', data=df, palette='viridis')
    
    # Customize the plot
    plt.title('Objective Value by Test Case', fontsize=16)
    plt.xlabel('Test Case', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(df['ObjectiveValue']):
        ax.text(i, v + 0.1, f"{v:.1f}", ha='center', fontsize=9)
    
    # Add average line
    avg_value = df['ObjectiveValue'].mean()
    plt.axhline(y=avg_value, color='r', linestyle='--', 
                label=f'Average: {avg_value:.1f}')
    
    # Add legend
    plt.legend()
    
    # Adjust layout to fit everything
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('objective_values.png')
    print("Plot saved as 'objective_values.png'")
    
    # Show the plot if not in non-interactive mode
    plt.show()

if __name__ == "__main__":
    main()
