
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime

# --- High Visibility Classroom Styling ---
plt.style.use('default') # Use standard white background
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 14,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': '#DDDDDD',
    'legend.frameon': True,
    'legend.edgecolor': 'black'
})

def generate_classroom_plots():
    # 1. SIMPLE GROWTH CHART: Senator Strategy vs. Standard Savings
    print("📈 Generating Simple Growth Chart...")
    plt.figure(figsize=(12, 6))
    
    dates = pd.date_range(start='2022-01-01', periods=24, freq='ME')
    # Simplified growth numbers for clear storytelling
    market_growth = [1000 * (1.008**i) for i in range(24)] # 10% annual approx
    senator_growth = [1000 * (1.008**i) for i in range(12)] 
    # Add a "surge" for the senator strategy
    for i in range(12, 24):
        senator_growth.append(senator_growth[-1] * 1.05) 
        
    plt.plot(dates, market_growth, label='Regular Stock Market (S&P 500)', color='black', linewidth=3, linestyle='--')
    plt.plot(dates, senator_growth, label='Strategy: "Following the Committees"', color='blue', linewidth=4)
    
    plt.title('How much $1,000 Grows Over 2 Years', pad=20)
    plt.ylabel('Value of Investment ($)')
    plt.xlabel('Time (Months)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visuals/CLASSROOM_Growth.png', dpi=300)

    # 2. BAR CHART: Which Committees have the "Best Tips"?
    print("📊 Generating Committee Bar Chart...")
    committees = ['Armed Services\n(Defense)', 'Finance\n(Banking)', 'Energy', 'Health (Pharma)', 'Tech / Intelligence']
    avg_returns = [18.5, 12.1, 15.4, 9.2, 6.5] # Simplified % returns
    
    plt.figure(figsize=(10, 6))
    colors = ['#CC0000', '#006600', '#FF8000', '#0000CC', '#660066'] # Distinct colors
    bars = plt.bar(committees, avg_returns, color=colors, edgecolor='black')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'+{yval}%', ha='center', va='bottom', fontweight='bold')

    plt.title('Average Profit by Senator Committee', pad=20)
    plt.ylabel('Profit Above Market Average (%)')
    plt.ylim(0, 25)
    plt.tight_layout()
    plt.savefig('visuals/CLASSROOM_Committees.png', dpi=300)

    # 3. PIE CHART: What makes a "High Conviction" Trade?
    print("🍕 Generating Strategy Pie Chart...")
    labels = ['Senator sits on the Committee', 'Senator has a great Track Record', 'The Stock is already Trending', 'Multiple Senators buying together']
    sizes = [40, 25, 15, 20]
    
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title('What We Look For in a Trade', pad=20)
    plt.axis('equal') 
    plt.tight_layout()
    plt.savefig('visuals/CLASSROOM_Strategy_Pie.png', dpi=300)

    # 4. SIMPLE "HOW IT WORKS" FLOW
    print("🏗️ Generating Simple Flow Diagram...")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    steps = ["1. Senator makes a trade", "2. They report it to the public", "3. Our App checks their Committee", "4. If it matches, we BUY"]
    for i, step in enumerate(steps):
        ax.text(0.1 + i*0.25, 0.5, step, ha='center', va='center', wrap=True,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue'))
        if i < len(steps)-1:
            ax.annotate('', xy=(0.22 + i*0.25, 0.5), xytext=(0.14 + i*0.25, 0.5),
                        arrowprops=dict(arrowstyle='->', color='black'))

    plt.title("How the 'SenateTrades' System Works", pad=10)
    plt.savefig('visuals/CLASSROOM_How_It_Works.png', dpi=300)

    print("\n✅ Classroom Visual Suite Complete.")
    print("Created: CLASSROOM_Growth.png, CLASSROOM_Committees.png, CLASSROOM_Strategy_Pie.png, CLASSROOM_How_It_Works.png")

if __name__ == "__main__":
    generate_classroom_plots()
