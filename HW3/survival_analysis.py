import pandas as pd
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

file_path = "C:/Users/user/Desktop/DS223_HW/HW3/telco.csv"
df = pd.read_csv(file_path)

print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())
print("\nFirst rows:")
print(df.head())

# encode for the survival analysis
binary_cols = ['internet', 'forward', 'churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes' : 1,'No' : 0})

categorical_cols = ['region', 'marital', 'custcat', 'gender', 'voice', 'retire', 'ed']
existing_cats = [col for col in categorical_cols if col in df.columns]

df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

print("\nAfter encoding shape:", df.shape)
print(df.dtypes)

duration_col = 'tenure'
event_col = 'churn'

outut_prefix = 'aft_results'

for c in df.columns:
    if df[c].dtype == bool:
        df[c] = df[c].astype(int)

covariates = [c for c in df.columns if c not in [duration_col, event_col]]
print("Number of covariates:", len(covariates))

# fitting AFT models

weib = WeibullAFTFitter()
lnorm = LogNormalAFTFitter()
llog = LogLogisticAFTFitter()

weib.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
lnorm.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
llog.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)

print("Weibull AIC:", weib.AIC_)
print("LogNormal AIC:", lnorm.AIC_)
print("LogLogistic AIC:", llog.AIC_)

# plotting the curves
baseline = df[covariates].mean().to_frame().T

weib_sf = weib.predict_survival_function(baseline)
lnorm_sf = lnorm.predict_survival_function(baseline)
llog_sf = llog.predict_survival_function(baseline)

plt.figure(figsize=(8,5))
plt.plot(weib_sf.index, weib_sf.values, label="Weibull")
plt.plot(lnorm_sf.index, lnorm_sf.values, label="LogNormal")
plt.plot(llog_sf.index, llog_sf.values, label="LogLogistic")

plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.title("Parametric AFT Model Survival Curves")
plt.legend()
plt.show()

summary = lnorm.summary
sig_features = summary[summary['p'] < 0.05].index.tolist()

sig_covariates = [c for c in sig_features if c in covariates]
print("Significant features:", sig_covariates)

final_model = lnorm

df["pred_lifetime"] = final_model.predict_expectation(df[covariates])
df["CLV"] = df["pred_lifetime"] * df["income"]


# exploring CLV within different customer segments

# segment by customer category
df["custcat_segment"] = "Basic service"
df.loc[df["custcat_E-service"] == 1, "custcat_segment"] = "E-service"
df.loc[df["custcat_Plus service"] == 1, "custcat_segment"] = "Plus service"
df.loc[df["custcat_Total service"] == 1, "custcat_segment"] = "Total service"

clv_cuscat = df.groupby("custcat_segment")["CLV"].mean()
print(clv_cuscat)

# segment by region
df["region_segment"] = "Zone 1"
df.loc[df["region_Zone 2"] == 1, "region_segment"] = "Zone 2"
df.loc[df["region_Zone 3"] == 1, "region_segment"] = "Zone 3"

clv_region = df.groupby("region_segment")["CLV"].mean()
print(clv_region)

# segment by gender
df["gender_segment"] = "Female"
df.loc[df["gender_Male"] == 1, "gender_segment"] = "Male"

clv_gender = df.groupby("gender_segment")["CLV"].mean()
print(clv_gender)

# identifying the at-risk customers
surv_12mo = final_model.predict_survival_function(df[covariates], times=[12]).T
df["surv_12mo"] = surv_12mo[12]
df["at_risk"] = df["surv_12mo"] < 0.5
retention_budget = df.loc[df["at_risk"], "CLV"].sum()
print("Estimated annual retention budget:", retention_budget)
