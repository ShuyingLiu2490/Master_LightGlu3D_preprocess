import numpy as np
import logging
import matplotlib.pyplot as plt

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_stats(arr):
    """Returns Min, Max, Median, Mean of a numpy array."""
    if len(arr) == 0:
        return 0, 0, 0, 0
    return np.min(arr), np.max(arr), np.median(arr), np.mean(arr)

def plot_sigma_distributions(sigma0_all, sigma1_all, num_queries=1, prefix=""):
    """Plots the distribution of the matchability (sigma) scores as Average Points per Query."""
    plt.figure(figsize=(14, 6))

    # Calculate statistics
    min0, max0, med0, mean0 = get_stats(sigma0_all)
    min1, max1, med1, mean1 = get_stats(sigma1_all)

    # Calculate weights to average the counts across the number of queries
    weights0 = np.ones_like(sigma0_all) / num_queries if len(sigma0_all) > 0 else []
    weights1 = np.ones_like(sigma1_all) / num_queries if len(sigma1_all) > 0 else []

    # score0 (2D points)
    plt.subplot(1, 2, 1)
    plt.hist(sigma0_all, bins=100, color='blue', alpha=0.7, range=(0, 1), weights=weights0)
    plt.title(f'Sigma0 (2D Keypoints) - {prefix}')
    plt.xlabel('Matchability (\u03c3)')
    plt.ylabel('Average Points per Query')
    plt.grid(True, alpha=0.3)
    
    # Add stats text box
    stats_text0 = f"Min: {min0:.4f}\nMax: {max0:.4f}\nMedian: {med0:.4f}\nMean: {mean0:.4f}"
    plt.gca().text(0.95, 0.95, stats_text0, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # score1 (3D points)
    plt.subplot(1, 2, 2)
    plt.hist(sigma1_all, bins=100, color='green', alpha=0.7, range=(0, 1), weights=weights1)
    plt.title(f'Sigma1 (3D Points) - {prefix}')
    plt.xlabel('Matchability (\u03c3)')
    plt.ylabel('Average Points per Query')
    plt.grid(True, alpha=0.3)
    
    # Add stats text box
    stats_text1 = f"Min: {min1:.4f}\nMax: {max1:.4f}\nMedian: {med1:.4f}\nMean: {mean1:.4f}"
    plt.gca().text(0.95, 0.95, stats_text1, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    # Use the prefix in the filename to prevent overwriting
    filename_suffix = f"_{prefix.lower()}" if prefix else ""
    filename = f"sigma_distributions{filename_suffix}.png"
    plt.savefig(filename, dpi=300)
    logger.info(f"Saved averaged sigma distribution plot to {filename}")

def show_results_with_sigma(all_sigma0, all_sigma1, num_queries=1, prefix=""):
    """Utility function to print stats and plot distributions."""
    # Process and Plot the Sigma Distributions
    if len(all_sigma0) > 0 and len(all_sigma1) > 0:
        logger.info(f"Aggregating {prefix} sigma scores for {num_queries} queries...")
        
        # Concatenate all the individual 1D arrays
        global_sigma0 = np.concatenate(all_sigma0)
        global_sigma1 = np.concatenate(all_sigma1)
        
        # Generate the histograms
        plot_sigma_distributions(global_sigma0, global_sigma1, num_queries=num_queries, prefix=prefix)
        
        # Log final stats to terminal
        min0, max0, med0, mean0 = get_stats(global_sigma0)
        min1, max1, med1, mean1 = get_stats(global_sigma1)
        
        logger.info(f"{prefix} Sigma 0 (2D) Statistics:")
        logger.info(f"Min: {min0:.4f} | Max: {max0:.4f} | Median: {med0:.4f} | Mean: {mean0:.4f}")
        logger.info(f"{prefix} Sigma 1 (3D) Statistics:")
        logger.info(f"Min: {min1:.4f} | Max: {max1:.4f} | Median: {med1:.4f} | Mean: {mean1:.4f}")

def _plot_with_fixed_limits(sigma0_all, sigma1_all, weights0, weights1, prefix, ylim0, ylim1):
    """Internal helper to plot histograms with locked Y-axis bounds."""
    plt.figure(figsize=(14, 6))

    min0, max0, med0, mean0 = get_stats(sigma0_all)
    min1, max1, med1, mean1 = get_stats(sigma1_all)

    # score0 (2D points)
    plt.subplot(1, 2, 1)
    plt.hist(sigma0_all, bins=100, color='blue', alpha=0.7, range=(0, 1), weights=weights0)
    plt.ylim(0, ylim0)  # lock y-axle scale based on day
    plt.title(f'Sigma0 (2D Keypoints) - {prefix}')
    plt.xlabel('Matchability (\u03c3)')
    plt.ylabel('Average Points per Query')
    plt.grid(True, alpha=0.3)
    
    stats_text0 = f"Min: {min0:.4f}\nMax: {max0:.4f}\nMedian: {med0:.4f}\nMean: {mean0:.4f}"
    plt.gca().text(0.95, 0.95, stats_text0, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # score1 (3D points)
    plt.subplot(1, 2, 2)
    plt.hist(sigma1_all, bins=100, color='green', alpha=0.7, range=(0, 1), weights=weights1)
    plt.ylim(0, ylim1)  # lock y-axle scale based on day
    plt.title(f'Sigma1 (3D Points) - {prefix}')
    plt.xlabel('Matchability (\u03c3)')
    plt.ylabel('Average Points per Query')
    plt.grid(True, alpha=0.3)
    
    stats_text1 = f"Min: {min1:.4f}\nMax: {max1:.4f}\nMedian: {med1:.4f}\nMean: {mean1:.4f}"
    plt.gca().text(0.95, 0.95, stats_text1, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    filename = f"sigma_distributions_{prefix.lower()}_fixed.png"
    plt.savefig(filename, dpi=300)
    logger.info(f"Saved FIXED SCALE sigma plot to {filename}")

def show_aachen_results_fixed_sigma(day_sigma0, day_sigma1, night_sigma0, night_sigma1, num_day, num_night):
    """Calculates the Y-axis scale from the Day queries and applies it to both Day and Night."""
    logger.info(f"Aggregating Aachen sigmas for {num_day} Day and {num_night} Night queries...")
    
    # Concatenate day
    d_s0 = np.concatenate(day_sigma0) if day_sigma0 else np.array([])
    d_s1 = np.concatenate(day_sigma1) if day_sigma1 else np.array([])
    w_d0 = np.ones_like(d_s0) / num_day if num_day > 0 else []
    w_d1 = np.ones_like(d_s1) / num_day if num_day > 0 else []
    
    # Calculate y limits based only on the day
    if len(d_s0) > 0 and len(d_s1) > 0:
        counts_d0, _ = np.histogram(d_s0, bins=100, range=(0, 1), weights=w_d0)
        counts_d1, _ = np.histogram(d_s1, bins=100, range=(0, 1), weights=w_d1)
        ylim0 = counts_d0.max() * 1.1 
        ylim1 = counts_d1.max() * 1.1
    else:
        ylim0, ylim1 = 100, 100 # Fallback if day is empty
        
    # Concatenate night
    n_s0 = np.concatenate(night_sigma0) if night_sigma0 else np.array([])
    n_s1 = np.concatenate(night_sigma1) if night_sigma1 else np.array([])
    w_n0 = np.ones_like(n_s0) / num_night if num_night > 0 else []
    w_n1 = np.ones_like(n_s1) / num_night if num_night > 0 else []
    
    # Plot both using the identical day limits
    if len(d_s0) > 0:
        _plot_with_fixed_limits(d_s0, d_s1, w_d0, w_d1, "Day", ylim0, ylim1)
    if len(n_s0) > 0:
        _plot_with_fixed_limits(n_s0, n_s1, w_n0, w_n1, "Night", ylim0, ylim1)