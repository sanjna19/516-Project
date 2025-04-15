import pandas as pd
from sklearn.preprocessing import LabelEncoder

def handle_missing_values(df, required_columns):
    """Drop rows with missing values in required columns."""
    return df.dropna(subset=required_columns)

def encode_columns(df, columns):
    """Label encode specified categorical columns."""
    le_dict = {}
    for col in columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

def bucket_age(df, age_column='Age'):
    """Create age group buckets."""
    def age_group(age):
        try:
            age = float(age)
        except:
            return 'Unknown'

        if age < 25:
            return '18-24'
        elif age < 35:
            return '25-34'
        elif age < 45:
            return '35-44'
        else:
            return '45+'

    df['age_group'] = df[age_column].apply(age_group)
    le = LabelEncoder()
    df['age_group_encoded'] = le.fit_transform(df['age_group'])
    return df, le