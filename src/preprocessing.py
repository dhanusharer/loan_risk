def feature_engineering(df):

    df = df.copy()

    df["loan_stress_index"] = df["loan_amnt"] / df["person_income"]
    df["credit_strength"] = df["credit_score"] * df["cb_person_cred_hist_length"]
    df["employment_stability"] = df["person_emp_exp"] / df["person_age"]

    return df
