import numpy as np
import pandas as pd
from prefect import task, flow

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow(name="Data Pipeline Flow")
def data_pipeline(arr):
    step_1 = create_series(arr)
    step_2 = clean_data(step_1)
    step_3 = summarize_data(step_2)
    return step_3

if __name__ == "__main__":
    summary_result = data_pipeline(arr)
    print("\nPrefect Pipeline Summary:")
    for key, value in summary_result.items():
        print(f"{key}: {value}")


# ====== Answers ======

# Q1: This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
# This  pipeline is very small and simple, setting up a full orchestration engine introduces extra code decorators, background logging, and state management that isn't really needed for such a tiny task.

# Q2: Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
# For example, when we need to run scripts on a regular time basis, or when we need error handling with automatic retries.
