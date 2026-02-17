"""
Project Name: Monte Carlo Simulation European Call Option for AAPL
================================================================================
Description:
            This code is a Monte Carlo Valuation Engine applying the Geometric Brownian Motion (GBM).
            Its purpose is to determine the "fair value" of a European Call Option
    by simulating 10,000 possible future timelines for a stock (like Apple).
    
The Setup (Defining the Market)
Asset: Apple Stock (S_0 = $255.78). 
Contract: A Call Option with a Strike Price of $265.00 expiring in 1 Year.
Risk-Free Rate (r = 4.1%)
Volatility (sigma = 28%), which dictate how the stock moves.

Usage:
    python3 Monte_Carlo Valuation.py

Author:  Mohit Jain
Created: [2026-01-15]
Updated: [2026-02-17]
"""

# Standard Library Imports
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

# --- 1. SETUP PARAMETERS ---
S0 = 255.78        # Current Stock Price of Apple Date_17.02.2026
K = 265.00         # Strike Price  - Out-of-the-Money (OTM) Call Option 
T = 1.0            # Time (1 Year)
r = 0.041          # Risk Free Rate - 1-Year US Treasury Bill Shrink to todays Value
sigma = 0.28       # Volatility - annualized standard deviation historical volatility hovers between 20% and 35%
num_simulations = 10000    # High number for accuracy
num_steps = 252            # Trading days
dt = T / num_steps         # Discrete Time Steps 

# Create Date Axis
start_date = date.today()
dates = [start_date + timedelta(days=i*(365/252)) for i in range(num_steps + 1)]

# --- 2. RUN SIMULATION (The Math) ---
# Generate random numbers
Z = np.random.standard_normal((num_simulations, num_steps))
S = np.zeros((num_simulations, num_steps + 1))
S[:, 0] = S0

# Calculate paths
for t in range(1, num_steps + 1):
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z[:, t-1]
    S[:, t] = S[:, t-1] * np.exp(drift + diffusion)

# --- 3. CALCULATE RESULTS ---
S_T = S[:, -1]
payoffs = np.maximum(S_T - K, 0)
option_price = np.mean(payoffs) * np.exp(-r * T)
prob_profit = np.sum(S_T > K) / num_simulations * 100
prob_loss = 100 - prob_profit

# --- 4. VISUALIZATION ---
# Create a layout: 2 Plots on top, Text area at bottom
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 2]) # Bottom row is for text

ax1 = fig.add_subplot(gs[0:2, 0]) # Left Plot (Spans 2 rows)
ax2 = fig.add_subplot(gs[0:2, 1]) # Right Plot (Spans 2 rows)
ax_text = fig.add_subplot(gs[2, :]) # Text Area (Spans full width at bottom)
ax_text.axis('off') # Hide axis for text area

# PLOT 1: The Paths (Squiggle Lines)
# Plot only first 50 paths to avoid clutter
ax1.plot(dates, S[:50, :].T, lw=1, alpha=0.5)
ax1.set_title(f'Monte Carlo Simulation: 50 Random Paths - Apple', fontsize=14, fontweight='bold')
ax1.set_ylabel('Stock Price ($)', fontsize=12)
ax1.set_xlabel('Date', fontsize=12)
ax1.axhline(y=K, color='black', linewidth=2, linestyle='--', label=f'Strike Price (${K})')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# PLOT 2: The Distribution (Histogram)
n, bins, patches = ax2.hist(S_T, bins=70, density=True, alpha=0.7, edgecolor='white')

# Color Logic: Green for Profit, Red for Loss
for bin_val, patch in zip(bins, patches):
    if bin_val >= K:
        patch.set_facecolor('forestgreen')
        patch.set_alpha(0.7)
    else:
        patch.set_facecolor('firebrick')
        patch.set_alpha(0.7)

ax2.set_title('Distribution of Final Prices at Maturity', fontsize=14, fontweight='bold')
ax2.set_xlabel('Price ($)', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.axvline(x=K, color='black', linewidth=2, linestyle='--', label='Strike Price')
ax2.axvline(x=np.mean(S_T), color='blue', linewidth=2, linestyle='-.', label=f'Avg Price (${np.mean(S_T):.0f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# TEXT: Formula & Definitions (Placed inside Left Plot for space efficiency)
formula_text = (
    r"$S_{t} = S_{t-1} \cdot e^{(r - \frac{1}{2}\sigma^2)dt + \sigma\sqrt{dt}Z}$" 
    "\n"
    r"$S_t$: Price | $r$: Rate | $\sigma$: Vol | $Z$: Random"
)
ax1.text(0.05, 0.05, formula_text, transform=ax1.transAxes,
         fontsize=11, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))

# TEXT: FINAL ANALYSIS REPORT
analysis_text = (
    f"SIMULATION RESULTS REPORT\n"
    f"--------------------------------------------------\n"
    f"Current Stock Price:   ${S0}\n"
    f"Target Strike Price:   ${K}\n"
    f"Fair Option Price:     ${option_price:.2f}\n"
    f"--------------------------------------------------\n"
    f"Probability of Profit (Green Area): {prob_profit:.1f}%\n\n"

)

ax_text.text(0.05, 0.9, analysis_text, fontsize=13, fontfamily='monospace', va='top')

plt.tight_layout()
plt.show()