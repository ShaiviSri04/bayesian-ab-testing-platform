import pandas as pd

def get_variant_distribution(df):
    #returns traffic allocation for each variant
    return(
        df["variant"].value_counts(normalize=True).reset_index())

def conversion_rates(df):
    #returns conversion rates by variant
    return(
        df.groupby("variant")["converted"].mean().reset_index()
    )

def get_ctr_rates(df):
    #returns click-through rates by variant
    return(
        df.groupby("variant")["clicked"].mean().reset_index()
    )

def get_revenue_metrics(df):
    #returns avg revenue per user by variant
    return(
        df.groupby("variant")["experiment_revenue"].mean().reset_index()
    )

def get_device_conversion(df):
    #returns conversion rates by device type and variant
    return(
        df.groupby(["variant","device_type"])["converted"].mean().reset_index()
    )

def get_segment_conversion(df):
    #returns conversion rates by customer segment and variant
    return(
        df.groupby(["variant","customer_segment"])["converted"].mean().reset_index()
    )