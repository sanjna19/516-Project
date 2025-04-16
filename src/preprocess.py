import pandas as pd
from sklearn.preprocessing import LabelEncoder
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing

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

def apply_reweighing(df, protected_attr, label_col):
    """
    Apply AIF360 Reweighing algorithm.
    Returns a DataFrame with an added 'instance_weight' column.
    """
    df_proc = df.copy()

    # Encode all object columns to ensure numeric-only input for AIF360
    for col in df_proc.columns:
        if df_proc[col].dtype == object:
            df_proc[col] = LabelEncoder().fit_transform(df_proc[col].astype(str))

    # Label encode protected_attr and label_col in original df as well
    le_prot = LabelEncoder()
    df[protected_attr] = le_prot.fit_transform(df[protected_attr].astype(str))
    df_proc[protected_attr] = df[protected_attr]

    le_label = LabelEncoder()
    df[label_col] = le_label.fit_transform(df[label_col].astype(str))
    df_proc[label_col] = df[label_col]

    # Retain only numeric columns for AIF360
    df_proc = df_proc.select_dtypes(include=['number'])

    privileged_value = df_proc[protected_attr].mode()[0]
    unprivileged_values = [v for v in df_proc[protected_attr].unique() if v != privileged_value]

    dataset = StandardDataset(
        df_proc,
        label_name=label_col,
        favorable_classes=[1],
        protected_attribute_names=[protected_attr],
        privileged_classes=[[privileged_value]],
        features_to_drop=[]
    )

    RW = Reweighing(
        unprivileged_groups=[{protected_attr: v} for v in unprivileged_values],
        privileged_groups=[{protected_attr: privileged_value}]
    )
    dataset_transf = RW.fit_transform(dataset)

    df['instance_weight'] = dataset_transf.instance_weights
    return df