import numpy as np
import pandas as pd

# Read the public test data
PUBLIC_TEST_FILE_PATH = "/Users/ian/Programming/ML/FlexTrack_Challenge/flextrack-challenge-2025-starter-kit/data/flextrack-2025-attempted-prediction-data-v0.1.csv"
public_test_df = pd.read_csv(PUBLIC_TEST_FILE_PATH)

# Generate random demand response flags
valid_demand_response_flags = [-1, 0, 1]
public_test_df["Demand_Response_Flag"] = np.random.choice(
    valid_demand_response_flags,
    size=len(public_test_df),
)

# Only keep the required columns for the evaluation of the submissions
public_test_df = public_test_df[["Site", "Timestamp_Local", "Demand_Response_Flag"]]

# Write the submission to a csv file
public_test_df.to_csv("/Users/ian/Programming/ML/FlexTrack_Challenge/flextrack-challenge-2025-starter-kit/data/my-submission-v0.1.csv", index=False)

# Submission the file to the challenge at: https://www.aicrowd.com/challenges/flextrack-challenge-2025/
# by clcking on the "Create Submission" button
# Note: You will have to click on "Participate" button and accept the terms and conditions of the challenge before you can make a submission.