#! /usr/bin/env python3
import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score

"""
This script aims to provide you a demonstration of how the scores are actually calculated. 
"""

########################################################
# Phase 1: Classification Task
########################################################

"""
Lets assume that we want to only focus on SiteA in the Training set, 
and want to calculate the scores for SiteA against some random predictions.
"""

# Load Training Data
training_df = pd.read_csv("data/flextrack-2025-training-data-v0.1.csv")
# Filter out only SiteA data
training_df = training_df[training_df["Site"] == "siteA"]

# Gather the ground truth demand response flags for siteA
ground_truth_demand_response_flags = training_df["Demand_Response_Flag"]

# Generate random predictions
random_predictions_demand_response_flags = np.random.choice(
    [-1, 0, 1],
    size=len(training_df),
)

# Calculate the geometric mean score
geometric_mean_score = geometric_mean_score(
    ground_truth_demand_response_flags,
    random_predictions_demand_response_flags,
    average="macro",
)

# Calculate the f1 score
f1_score = f1_score(
    ground_truth_demand_response_flags,
    random_predictions_demand_response_flags,
    average="macro",
)

print(f"Geometric Mean Score: {geometric_mean_score}")
print(f"F1 Score: {f1_score}")


########################################################
# Phase 2: Regression Task
########################################################

"""
TODO: The Phase 2 evaluation metrics will be released during the launch of the Phase 2 of this competition
"""