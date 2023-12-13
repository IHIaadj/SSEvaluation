import numpy as np
from utils import * 

def calculate_pds(jacob_cov_values, accuracy_values):
    """
    Calculate the Performance Diversity Score (PDS).

    Parameters:
    jacob_cov_values (list): List of Jacobian covariance values for each architecture.
    accuracy_values (list): List of accuracy values for each architecture.

    Returns:
    float: The calculated Performance Diversity Score.
    """
    sd_jacob_cov = np.std(jacob_cov_values)
    sd_accuracy = np.std(accuracy_values)
    sd_pooled = (sd_jacob_cov + sd_accuracy) / 2

    pds = abs(sd_jacob_cov - sd_accuracy) / sd_pooled
    return pds

# Example usage
jacob_cov_values = [0.1, 0.2, 0.15, 0.3]  # Replace with your actual Jacobian covariance values
accuracy_values = [0.9, 0.85, 0.88, 0.92]  # Replace with your actual accuracy values

pds_score = calculate_pds(jacob_cov_values, accuracy_values)
print(f"PDS Score: {pds_score}")
