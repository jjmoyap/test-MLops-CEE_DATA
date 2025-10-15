from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def combine_rare(df, col, threshold=0.2):
    counts = df[col].value_counts(normalize=True)
    rare = counts[counts < threshold].index
    df[col] = df[col].replace(rare, 'OTHERS')
    return df

def create_ordinal_features(df, ordinal_cols, ord_map):
    ord_enc = OrdinalEncoder(categories=[ord_map]*len(ordinal_cols))
    df[[col+'_num' for col in ordinal_cols]] = ord_enc.fit_transform(df[ordinal_cols])
    df['Academic_Score'] = df[[col+'_num' for col in ordinal_cols]].mean(axis=1)
    return df

def add_frequency_features(df, categorical_cols):
    for col in categorical_cols:
        df[col+'_freq'] = df[col].map(df[col].value_counts(normalize=True))
        target_mean = df.groupby(col)['Performance_num'].mean()
        df[col+'_target_mean'] = df[col].map(target_mean)
    return df
