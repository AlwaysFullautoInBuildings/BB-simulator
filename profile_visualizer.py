import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_profile():
    # Read the CSV file
    df = pd.read_csv('profile_log.csv')
    
    # Sort by cumulative time and get top 20 functions
    df_sorted = df.nlargest(20, 'Cumulative Time')
    
    # Set up the plot style
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), facecolor='#1C1C1C')
    fig.suptitle('Profiler Results Analysis', color='white', size=16)
    
    # Adjust spacing between title and plots
    plt.subplots_adjust(top=0.9)
    
    # Plot 1: Cumulative Time
    sns.barplot(data=df_sorted, 
                x='Cumulative Time', 
                y='Function',
                ax=ax1,
                palette='viridis')
    ax1.set_title('Top 20 Functions by Cumulative Time', color='white', pad=10)
    ax1.set_xlabel('Cumulative Time (seconds)', color='white')
    ax1.set_ylabel('Function', color='white')
    ax1.tick_params(colors='white')
    
    # Plot 2: Calls Count
    sns.barplot(data=df_sorted,
                x='Total Calls',
                y='Function',
                ax=ax2,
                palette='viridis')
    ax2.set_title('Number of Calls for Top Functions', color='white', pad=10)
    ax2.set_xlabel('Number of Calls', color='white')
    ax2.set_ylabel('Function', color='white')
    ax2.tick_params(colors='white')
    
    # Adjust layout and display
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space for suptitle
    for ax in [ax1, ax2]:
        ax.set_facecolor('#2F2F2F')
    fig.patch.set_facecolor('#1C1C1C')
    
    # Add some stats as text
    total_time = df['Cumulative Time'].sum()
    total_calls = df['Total Calls'].sum()
    stats_text = f'Total Runtime: {total_time:.2f}s\nTotal Function Calls: {total_calls:,}'
    fig.text(0.02, 0.98, stats_text, color='white', fontsize=10, ha='left', va='top')
    
    plt.show()

if __name__ == "__main__":
    try:
        visualize_profile()
    except FileNotFoundError:
        print("Error: profile_log.csv not found. Run the simulator first to generate profiling data.")
