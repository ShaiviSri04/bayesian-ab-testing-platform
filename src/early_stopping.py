import pandas as pd
from src.bayesian import probability_b_wins

def confidence_curve(df):
    traffic_checkpoints=[
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        1.00
    ]

    winning_probabilities=[]
    for percent in traffic_checkpoints:
        sample_size=int(len(df)*percent)

        partial_df=df.sample(
            n=sample_size,
            random_state=42
        )

        prob_b_better=probability_b_wins(partial_df)
        winning_probabilities.append(prob_b_better)

    results_df=pd.DataFrame({
        "traffic_pct":[
            pct*100 for pct in traffic_checkpoints
        ],
        "prob_b_better": winning_probabilities
    })
    return results_df

def find_early_stop(df,threshold=0.95):
    results_df=confidence_curve(df)
    winners=results_df[results_df["prob_b_better"]>=threshold]

    if len(winners)==0:
        return None
    
    early_stop=winners.iloc[0]["traffic_pct"] #smallest traffic with 95% confidence
    return early_stop

def users_saved(df, threshold=0.95):

    early_stop = find_early_stop(df, threshold)

    print("EARLY STOP:", early_stop)

    if early_stop is None:
        return 0

    total_users = len(df)

    users_used = int(total_users * (early_stop / 100))

    saved_users = total_users - users_used

    print("TOTAL:", total_users)
    print("USED:", users_used)
    print("SAVED:", saved_users)

    return saved_users