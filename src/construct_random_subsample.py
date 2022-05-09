import pandas as pd


def construct_random_subsample_of_collision_data(df: pd.DataFrame) -> pd.DataFrame:
    serious_df = df[df["serious"] == 1]
    non_serious_df = df[df["serious"] == 0]
    serious_count = serious_df["serious"].count()
    # Using the same random seed to ensure repeatability across runs.
    new_df = pd.concat([serious_df, non_serious_df.sample(n=serious_count, random_state=42)])
    new_df = new_df.sample(frac=1).reset_index(drop=True)
    return new_df
