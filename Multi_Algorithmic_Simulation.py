"""
Project: Multi-Algorithmic Simulation of Derivative Structures and Valuation of Index Options.
         European (DAX), American (S&P 500), Asian (Nikkei 225)

Author Mohit - feel free to discuss the Topic or simply connect.
Linkedin:[https://www.linkedin.com/in/mohit-jain-3a5206342/]
GitHub:  [https://github.com/MohitJain-git-source]

Current Version V3
V1 [2026-01-17] - European Call Option Monte Carlo
V2 [2026-02-20] - Dynamic Approach yfinance
V3 [2026-02-21] - Added American, Asian Call Options, Summary Evaluation
V3 [2026-02-21] - Monte Carlo Algorithm (Standard, Average) and FDM for American Option, PuT Option

Future Add ons:
1. Change of Asset.
2. Gui Interface for Simulation Input adjustments

Description:
    This code is a multi-algorithmic Valuation Engine.
    It applies specific numerical methods to match option structures:
    1. European Option: DAX Index (^GDAXI) -> Standard Monte Carlo
    2. American Option: S&P 500 (^GSPC) -> Implicit Finite Difference Method (FDM)
    3. Asian Option: Nikkei 225 (^N225) -> Path-Averaging Monte Carlo
    4. It simulates 10 000 paths and compares Volatility, Probability of Profit (POP) and
       Option Premium vs. Spot Price of each Structure and Index.

Break Down of Code Sections:
    1. Import libaries and Define Directory for Results.
    2. Execute Function simulate_plot - asks the User if Put or  Call wants to be Simulated
     2.1 Dynamically fetch current price and 2-year historical volatility
         using the yfinance library (Yahoo Finance). (505 Day - 252 Trading Days per year)
     2.2 Calculate the logaritgmic returns and Standard Deviation
     2.3 Annualize th daily Volatility into sigma.
     2.4 Setup Parameter for Simulation: Strike Price (K), Time (T), Risk-Free Rate (r),
     2.5 Run Simulation (Geometric Brownian Motion) - The Loop calculates the daily price movement
         and calculates the Probability of Profit (POP)
     2.6 Selection of correct Algorithm
         (European  - standard Monte Carlo)
         (Asian     - Path Averaging Monte Carlo)
         (American  - Implicid Finite Difference  Method (FDM)/(Black Scholes Partial Differential Equation)
     2.7 Visualization via Matplotlib of each Version.
    3. iIt also generates a summary comparison chart of key metrics across all three regions.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import yfinance as yf
from datetime import date, timedelta

# --- DIRECTORY SETUP adjust if not defined ---
ASSETS_DIR = "assets"
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)


def simulate_and_plot(region, option_style, ticker_symbol, option_type):
    print(
        f"Fetching data and running simulation for {option_style} {option_type} Option ({region}: {ticker_symbol})...")

    # --- 1. DYNAMIC DATA RETRIEVAL ---
    asset = yf.Ticker(ticker_symbol)
    hist = asset.history(period="3y")
    close_prices = hist['Close'].tail(505)

    S0 = float(close_prices.iloc[-1])
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    sigma = log_returns.std() * np.sqrt(252)

    # --- 2. SETUP PARAMETERS ---
    # Since index prices vary greatly, we dynamically set the Strike Price (K)
    # to be 5% Out-of-The-Money (OTM) for a Call or Put Option.
    if option_type == "Call":
        K = S0 * 1.05
        otm_label = "+5% OTM"
    else:
        K = S0 * 0.95
        otm_label = "-5% OTM"

    T = 1.0  # Time (1 Year)
    r = 0.041  # Risk Free Rate - 1-Year US Treasury Bill approx.
    num_simulations = 10000  # High number for accuracy
    num_steps = 252  # Trading days in the simulated year
    dt = T / num_steps

    start_date = date.today()
    dates = [start_date + timedelta(days=i * (365 / 252)) for i in range(num_steps + 1)]

    # --- 3. RUN SIMULATION FOR PATHS (Used by all for visualization & POP) ---
    Z = np.random.standard_normal((num_simulations, num_steps))
    S = np.zeros((num_simulations, num_steps + 1))
    S[:, 0] = S0

    for t in range(1, num_steps + 1):
        drift = (r - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z[:, t - 1]
        S[:, t] = S[:, t - 1] * np.exp(drift + diffusion)

    S_T = S[:, -1]

    # Calculate Probability of Profit (POP) using the simulated paths
    if option_type == "Call":
        prob_profit = np.sum(S_T > K) / num_simulations * 100
    else:
        prob_profit = np.sum(S_T < K) / num_simulations * 100

    # --- 4. ALGORITHM SELECTION FOR PRICING ---
    if option_style == "European":
        # Standard Terminal Payoff
        if option_type == "Call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)

        option_price = np.mean(payoffs) * np.exp(-r * T)
        calc_method = "Standard Monte Carlo"

    elif option_style == "Asian":
        # Path-Dependent Average Payoff
        S_mean = np.mean(S, axis=1)
        if option_type == "Call":
            payoffs = np.maximum(S_mean - K, 0)
        else:
            payoffs = np.maximum(K - S_mean, 0)

        option_price = np.mean(payoffs) * np.exp(-r * T)
        calc_method = "Path-Averaging Monte Carlo"

    elif option_style == "American":
        """
        ========================================================================
        IMPLICIT FINITE DIFFERENCE METHOD (FDM) FOR AMERICAN OPTIONS
        ======================================================================== 
        FDM solves the Black-Scholes Partial Differential Equation (PDE) by creating a 
        grid of Prices (M) and Time (N), and stepping *backward* from expiration to today.
        The 'Implicit' method is used because it is unconditionally mathematically stable, 
        unlike the 'Explicit' method which can blow up if the time steps aren't tiny enough.

        The algorithm first builds a grid of possible stock prices and time steps, 
        calculating the option's known payoff at the final expiration date. 
        It then steps backward through time, solving a matrix at each step to determine 
        the option's theoretical holding value based on volatility, interest rates, and time decay. 
        Finally, at every single point on that backward-looking grid, it compares this computed 
        holding value against the immediate profit of exercising early, always keeping the 
         higher number to perfectly capture the American early-exercise premium.
        ========================================================================
        """

        # Implicit Finite Difference Method
        M = 150  # Price steps
        N = 1000  # Time steps
        S_max = S0 * 3.0
        dt_fdm = T / N

        S_grid = np.linspace(0, S_max, M + 1)
        if option_type == "Call":
            V = np.maximum(S_grid - K, 0)
        else:
            V = np.maximum(K - S_grid, 0)

        # Tridiagonal Matrix Coefficients
        j = np.arange(1, M)
        q = 0.015  # 1.5% dividend yield to trigger early-exercise

        alpha = 0.5 * dt_fdm * (sigma ** 2 * j ** 2 - (r - q) * j)
        beta = 1 + dt_fdm * (sigma ** 2 * j ** 2 + r)
        gamma = 0.5 * dt_fdm * (sigma ** 2 * j ** 2 + (r - q) * j)

        ab = np.zeros((3, M - 1))
        ab[0, 1:] = -gamma[:-1]  # Upper
        ab[1, :] = beta  # Main
        ab[2, :-1] = -alpha[1:]  # Lower

        for i in range(N):
            B = V[1:M].copy()
            # Boundary Conditions
            if option_type == "Call":
                V[0] = 0
                V[M] = S_max - K * np.exp(-r * (i + 1) * dt_fdm)
            else:
                V[0] = K * np.exp(-r * (i + 1) * dt_fdm)
                V[M] = 0

            B[0] += alpha[0] * V[0]
            B[-1] += gamma[-1] * V[M]

            # Solve System
            V[1:M] = linalg.solve_banded((1, 1), ab, B)

            # Apply American Early-Exercise Condition
            if option_type == "Call":
                V = np.maximum(V, S_grid - K)
            else:
                V = np.maximum(V, K - S_grid)

        option_price = np.interp(S0, S_grid, V)
        calc_method = "Implicit Finite Difference"

    # --- 5. VISUALIZATION ---
    fig = plt.figure(figsize=(16, 10))
    try:
        fig.canvas.manager.set_window_title(f"{option_style} {option_type} Option - {region}")
    except AttributeError:
        pass

    gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 2])

    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')

    # PLOT 1: The Paths
    ax1.plot(dates, S[:50, :].T, lw=1, alpha=0.5)
    ax1.set_title(f'Simulated Paths - {region} ({ticker_symbol})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Index Price', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.axhline(y=K, color='black', linewidth=2, linestyle='--', label=f'Strike Price ({K:,.2f})')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # PLOT 2: The Distribution
    n, bins, patches = ax2.hist(S_T, bins=70, density=True, alpha=0.7, edgecolor='white')
    for bin_val, patch in zip(bins, patches):
        if option_type == "Call":
            patch.set_facecolor('forestgreen' if bin_val >= K else 'firebrick')
        else:
            patch.set_facecolor('forestgreen' if bin_val <= K else 'firebrick')

    ax2.set_title(f'Final Price Distribution (2-Yr Vol: {sigma:.2%})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Price', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.axvline(x=K, color='black', linewidth=2, linestyle='--', label='Strike Price')
    ax2.axvline(x=np.mean(S_T), color='blue', linewidth=2, linestyle='-.', label=f'Avg Price ({np.mean(S_T):,.0f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # FINAL ANALYSIS REPORT
    analysis_text = (
        f"SIMULATION RESULTS: {option_style.upper()} {option_type.upper()} OPTION PREVIEW\n"
        f"--------------------------------------------------\n"
        f"Asset:                 {region} ({ticker_symbol})\n"
        f"Current Price (S0):    {S0:,.2f}\n"
        f"Target Strike (K):     {K:,.2f} ({otm_label})\n"
        f"2-Year Volatility:     {sigma:.2%}\n"
        f"Fair Option Price:     {option_price:,.2f} (Using {calc_method})\n"
        f"--------------------------------------------------\n"
        f"Probability of Profit: {prob_profit:.1f}%\n"
    )
    ax_text.text(0.05, 0.9, analysis_text, fontsize=13, fontfamily='monospace', va='top')
    plt.tight_layout()

    safe_region_name = region.replace(" ", "_")
    plot_filepath = os.path.join(ASSETS_DIR, f"{option_style}_{option_type}_{safe_region_name}.png")
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_filepath}")

    return {
        "Region": region,
        "Volatility": sigma,
        "Probable_Profit": prob_profit,
        "Relative_Premium": (option_price / S0) * 100
    }


def plot_summary(results, option_type):
    """
       Creates a 1x3 subplot figure comparing Volatility, Probability of Profit,
       and Relative Premium across all simulated regions, and saves it in assets.
    """
    regions = [r["Region"] for r in results]
    vols = [r["Volatility"] * 100 for r in results]
    probs = [r["Probable_Profit"] for r in results]
    prems = [r["Relative_Premium"] for r in results]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    try:
        fig.canvas.manager.set_window_title(f"Summary Comparison: {option_type} Options")
    except AttributeError:
        pass

    fig.suptitle(f'Option Pricing Summary Comparison (5% OTM {option_type}, 1-Year)', fontsize=16, fontweight='bold')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 1. Volatility Plot
    bars0 = axs[0].bar(regions, vols, color=colors, alpha=0.8, edgecolor='black')
    axs[0].set_title('2-Year Annualized Volatility', fontsize=12)
    axs[0].set_ylabel('Volatility (%)', fontsize=11)
    axs[0].grid(axis='y', alpha=0.3)
    for bar in bars0:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom',
                    fontweight='bold')

    # 2. Probability of Profit Plot
    bars1 = axs[1].bar(regions, probs, color=colors, alpha=0.8, edgecolor='black')
    axs[1].set_title('Probability of Profit', fontsize=12)
    axs[1].set_ylabel('Probability (%)', fontsize=11)
    axs[1].grid(axis='y', alpha=0.3)
    for bar in bars1:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom',
                    fontweight='bold')

    # 3. Relative Premium Plot
    bars2 = axs[2].bar(regions, prems, color=colors, alpha=0.8, edgecolor='black')
    axs[2].set_title('Option Premium vs. Spot Price', fontsize=12)
    axs[2].set_ylabel('Premium (% of Spot)', fontsize=11)
    axs[2].grid(axis='y', alpha=0.3)
    for bar in bars2:
        yval = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.2f}%", ha='center', va='bottom',
                    fontweight='bold')

    plt.tight_layout()
    summary_filepath = os.path.join(ASSETS_DIR, f"Summary_Comparison_{option_type}.png")
    plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_filepath}")


# Runs the Simulation with the Given Input (Configs)
if __name__ == "__main__":
    configs = [
        {"region": "DAX Index", "style": "European", "ticker": "^GDAXI"},
        {"region": "S&P 500", "style": "American", "ticker": "^GSPC"},
        {"region": "Nikkei 225", "style": "Asian", "ticker": "^N225"}
    ]

    while True:
        print("\n--- NEW SIMULATION RUN ---")
        user_choice = ""
        while user_choice not in ["Call", "Put"]:
            user_choice = input("Would you like to simulate 'Call' or 'Put' options? ").strip().capitalize()
            if user_choice not in ["Call", "Put"]:
                print("Invalid input. Please type exactly 'Call' or 'Put'.")

        results_data = [simulate_and_plot(cfg["region"], cfg["style"], cfg["ticker"], user_choice) for cfg in configs]
        plot_summary(results_data, user_choice)

        print("\nSimulations complete. Displaying plots... (Close the plot windows to continue)")
        plt.show()

        run_again = input("\nWould you like to run another simulation? (yes/no): ").strip().lower()
        if run_again not in ['yes', 'y']:
            print("Exiting simulator. Have a great day!")
            break