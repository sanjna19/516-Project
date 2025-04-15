
def selection_rate(df, group, group_col, label_col='Employment'):
    total = df[df[group_col] == group].shape[0]
    selected = df[(df[group_col] == group) & (df[label_col] == 1)].shape[0]
    return selected / total if total > 0 else 0

def print_group_rates(df, group_col):
    print(f"\nSelection Rates by {group_col}:")
    for group in df[group_col].unique():
        rate = selection_rate(df, group, group_col)
        print(f"{group}: {rate:.2f}")

def disparate_impact(df, unprivileged, privileged, group_col):
    sr_unpriv = selection_rate(df, unprivileged, group_col)
    sr_priv = selection_rate(df, privileged, group_col)
    ratio = sr_unpriv / sr_priv if sr_priv > 0 else 0
    print(f"\nDisparate Impact ({unprivileged}/{privileged}): {ratio:.2f}")
    return ratio

def equal_opportunity(y_true, y_pred, group_values, group_col, df):
    print("\nEqual Opportunity by group:")
    for group in group_values:
        group_idx = df[group_col] == group
        tp = ((y_true[group_idx]) & (y_pred[group_idx] == 1)).sum()
        pos = y_true[group_idx].sum()
        rate = tp / pos if pos > 0 else 0
        print(f"{group}: TPR = {rate:.2f}")