import logging as log
import os
import pandas as pd
from train_test_split import train_test_split

def combine_df(df, df_result, df_key, df_result_key, df_result_cas_key):
    value_list = df[df_key].values
    df_result_list = df_result[df_result_key].values

    for val in value_list:
        if val in df_result_list:

            df.loc[df[df_key] == val, "Seriousness"] = str(df_result[df_result_cas_key].loc[df_result[df_result_key] == val].values[0]).lower()

        else:
            df.loc[df[df_key] == val, "Seriousness"] = "0 unknown"

    return df

def servity_map(df):
    serverity_status_map = {'3 slight': 0, '1 fatal': 1, '2 serious': 1}
    df['Seriousness'] = df['Seriousness'].map(serverity_status_map)


def preprocess_time(val):
    if val == None:

        return 25

    re_val = ""
    if val[0] == "'":
        re_val += val[1]
        re_val += val[2]
    else:
        val_list = val.split(":")
        re_val += val_list[0]

    return re_val

def preprocess_date(val):

    if val == None:

        return 0

    if int(val[-1]) < 9 and int(val[-1]) > 0:

        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        val_list = val.split("-")
        re_val = month_map[val_list[1]]

    else:
        val_list = val.split("/")
        re_val = val_list[1]

    return int(re_val)

def fill_na_df(df):

    for col in df:
        #get dtype for column
        dt = df[col].dtype
        #check if it is a number
        if dt == "int64" or dt == "float64":
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna("Null", inplace=True)

def load_data():

    # First get the `$PROJECT_ROOT/data` path.
    data_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    log.info(f"trying to load data from: `{data_path}`")
    df_2020 = pd.read_csv("../data/2020/2020-gla-data-extract-attendant.csv")
    df_2019 = pd.read_csv("../data/2019/2019-gla-data-extract-attendant.csv")
    df_2018 = pd.read_csv("../data/2018/2018-data-files-attendant.csv")
    df_2017 = pd.read_csv("../data/2017/2017-data-attendant.csv")
    df_2016 = pd.read_csv("../data/2016/2016-gla-data-extract-attendant.csv")
    df_2015 = pd.read_csv("../data/2015/2015-gla-data-extract-attendant.csv")

    df_result_2020 = pd.read_csv("../data/2020/2020-gla-data-extract-casualty.csv")
    df_result_2019 = pd.read_csv("../data/2019/2019-gla-data-extract-casualty.csv")
    df_result_2018 = pd.read_csv("../data/2018/2018-data-files-casualty.csv")
    df_result_2017 = pd.read_csv("../data/2017/2017-data-casualty.csv")
    df_result_2016 = pd.read_csv("../data/2016/2016-gla-data-extract-casualty.csv")
    df_result_2015 = pd.read_csv("../data/2015/2015-gla-data-extract-casualty.csv")

    # del df_2020["ADATES_FULL"]
    # del df_2020["APOLICER_DECODED"]
    #
    # del df_2019["ADATES_FULL"]
    # del df_2019["APOLICER_DECODED"]

    combine_df(df_2020, df_result_2020, "Accident Ref", "Accident Ref.", "Casualty Severity")
    combine_df(df_2019, df_result_2019, "Accident Ref", "AREFNO", "Casualty Severity")
    combine_df(df_2018, df_result_2018, "Accident Ref.", "Accident Ref.", "Casualty Severity")
    combine_df(df_2017, df_result_2017, "Accident Ref.", "Accident Ref.", "Casualty Severity")
    combine_df(df_2016, df_result_2016, "Accident Ref.", "Accident Ref.", "Casualty Severity")
    combine_df(df_2015, df_result_2015, "Accident Ref.", "Accident Ref.", "Casualty Severity")
    frame = [df_2015, df_2016, df_2017, df_2018, df_2019, df_2020]
    # frame = [df_2015, df_2016, df_2017, df_2018, df_2019]
    # frame = [df_2019, df_2020]
    frame = [df_2015, df_2016]

    df = pd.concat(frame, ignore_index=True)
    print("Finish concatenate all dfs")
    servity_map(df)
    print("Finish Preprocessing Servity")
    # df["Time"] = df["Time"].apply(preprocess_time)
    del df["Time"]
    print("Finish Preprocessing Time")

    # df["Accident Date"] = df["Accident Date"].apply(preprocess_date)
    del df["Accident Date"]
    print("Finish Preprocessing Accident Date")

    # df = df.drop("Accident Severity", axis=1)
    del df["Accident Severity"]
    print("Finish Preprocessing Df")


    # fill_na_df(df)
    # print("Finish Fill Nan")

    return df

df = load_data()
df.drop("Accident Ref.")
df.dropna(inplace=True)
categorical_columns = df.select_dtypes(exclude = 'number').drop('Seriousness', axis = 1).columns
df.info()
# train_count = [0,0,0,0]
# test_count = [0,0,0,0]


# y = df.pop("Seriousness")
# X = df.drop("Location", axis=1)

# counter = 0

# X_train, X_test, y_train, y_test = train_test_split(X,y)
#
# print(y_test.head())

# for (idx, row) in df.iterrows():
#     if counter <= 102687:
#         train_count[int(row.loc["Seriousness"])] += 1
#     else:
#         test_count[int(row.loc["Seriousness"])] += 1
#
#     counter += 1
# #
# # for (idx, row) in y_test.iterrows():
# #     test_count[int(row.loc["Seriousness"])] += 1
# #
# print("Traing analysing")
# print(train_count)
#
# print("Test analysing")
# print(test_count)