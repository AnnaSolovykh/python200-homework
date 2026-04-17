import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Task1 ---
df = pd.read_csv("resources/student_performance_math.csv", sep=";")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

plt.clf()
plt.hist(df["G3"], bins=21, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3 (final grade)")
plt.ylabel("Count")
plt.savefig("assignments_02/outputs/g3_distribution.png")

# --- Task 2 ---
#  Add a comment explaining your reasoning -- why would keeping these rows distort the model?

# G3 = 0 here means not a real score of 0, but presumably the student did not take the final exam.
# Keeping those rows mixes two different outcomes - missing a grade vs. a real grade,
# which makes the model biased.

print("\nShape before removing G3 == 0:", df.shape)
df_model = df[df["G3"] != 0].copy()
print("Shape after removing G3 == 0:", df_model.shape)

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df_model[col] = (df_model[col] == "yes").astype(int)
df_model["sex"] = (df_model["sex"] == "M").astype(int)

corr_orig = df["absences"].corr(df["G3"])
corr_filt = df_model["absences"].corr(df_model["G3"])
print("\nPearson correlation (absences, G3) — full data:", corr_orig)
print("Pearson correlation (absences, G3) — filtered:", corr_filt)

# Add a comment explaining why filtering changes the result: what were students with G3=0 doing in the original data that made absences look like a weak predictor? 

# In this dataset, students with G3 = 0 all have absences = 0, so they add many
# identical points at (0, 0). That large cluster at the origin weakens the linear
# relationship between absences and G3 among students who actually received a grade.


plt.clf()
plt.scatter(df["absences"], df["G3"], alpha=0.5)
plt.xlabel("Absences")
plt.ylabel("G3")
plt.title("Full data (includes G3 = 0)")
plt.savefig("assignments_02/outputs/absences_vs_g3_full.png")

plt.clf()
plt.scatter(df_model["absences"], df_model["G3"], alpha=0.5)
plt.xlabel("Absences")
plt.ylabel("G3")
plt.title("Filtered (G3 > 0 only)")
plt.savefig("assignments_02/outputs/absences_vs_g3_filtered.png")


# --- Task 3 ---
# Drop zero-variance numeric columns: Pearson r is undefined if a feature is constant.
numeric_predictors = [
    c
    for c in df_model.select_dtypes(include="number").columns
    if c not in ("G1", "G2", "G3") and df_model[c].std() != 0
]

corr_with_g3 = pd.Series(
    {c: df_model[c].corr(df_model["G3"]) for c in numeric_predictors},
    dtype=float,
).sort_values()
print("\nPearson correlation with G3 (most negative → most positive):")
print(corr_with_g3)
strongest = corr_with_g3.abs().idxmax()
print("\nStrongest linear relationship with G3 (by |r|):", strongest, "=", corr_with_g3[strongest])

# Which feature has the strongest relationship with G3? 
# Are any results surprising?

# At first glance, it's surprising that extra school support has the second-strongest
# negative correlation with G3. But it makes sense as remediation is usually given to lower-performing
# students, so the variable marks struggling learners rather than causing worse grades.

# EDA plots guided by correlation strengths: Medu and failures.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

df_model.boxplot(column="G3", by="Medu", ax=axes[0], grid=False)
fig.suptitle("")
axes[0].set_title("G3 by mother's education (Medu)")
axes[0].set_xlabel("Medu (0=none … 4=higher)")
axes[0].set_ylabel("G3")
# Higher maternal education tracks with higher median final grades, though boxes overlap.

df_model.boxplot(column="G3", by="failures", ax=axes[1], grid=False)
fig.suptitle("")
axes[1].set_title("G3 by past class failures")
axes[1].set_xlabel("Failures (0–3, capped)")
axes[1].set_ylabel("G3")
# More past failures shifts the whole grade distribution down.

fig.savefig("assignments_02/outputs/g3_medu_failures_eda.png")

# --- Task 4 ---
x_base = df_model[["failures"]]
y_base = df_model["G3"]
x_train, x_test, y_train, y_test = train_test_split(
    x_base, y_base, test_size=0.2, random_state=42
)

baseline = LinearRegression()
baseline.fit(x_train, y_train)
y_pred_base = baseline.predict(x_test)
rmse_base = np.sqrt(np.mean((y_test - y_pred_base) ** 2))
r2_base = baseline.score(x_test, y_test)

print("\nBaseline (failures only):")
print("Slope:", baseline.coef_[0])
print("RMSE (test):", rmse_base)
print("R² (test):", r2_base)

# Add a comment: given that grades are on a 0-20 scale, what do the slopes and RMSE tell you in plain English?
# Is R² better or worse than you expected from exploratory data analysis?

# G3 is 0–20, so the slope is “how many points the grade moves” when failures goes up by 1 (here ~−1.4 per step).
# RMSE is also in grade points—mine is ~3, so guesses are often wrong by a few points, which feels normal with one feature.
# R² ~ 0.09 is pretty low; that fits my EDA (r ~ −0.29 → r² ~ 0.08). I didn’t expect miracles from failures alone.

# --- Task 5 ---
df_clean = df_model  

feature_cols = [
    "failures",
    "Medu",
    "Fedu",
    "studytime",
    "higher",
    "schoolsup",
    "internet",
    "sex",
    "freetime",
    "activities",
    "traveltime",
]
x = df_clean[feature_cols].values
y = df_clean["G3"].values

x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train_f, y_train_f)
y_pred_f = model.predict(x_test_f)

r2_train_f = model.score(x_train_f, y_train_f)
r2_test_f = model.score(x_test_f, y_test_f)
rmse_f = np.sqrt(np.mean((y_test_f - y_pred_f) ** 2))

print("\nFull model (all features):")
print("R² (train):", r2_train_f)
print("R² (test):", r2_test_f)
print("RMSE (test):", rmse_f)
print("Test R² vs baseline (Task 4):", r2_test_f, "vs", r2_base, "→ delta:", r2_test_f - r2_base)

print("\nCoefficients:")
for i in range(len(feature_cols)):
    print(feature_cols[i], ":", round(model.coef_[i], 3))

# Look at the coefs (sort big → small in your head). Any surprising + / − signs?
# On my output the biggest positives are things like internet, higher, studytime, Fedu, Medu, sex that mostly matches what I'd expect.
# The weirdest one is schoolsup again (strong minus) but, as I said in Task 3, support targets weaker students.

# Train R² vs test R² — close or a big gap? What does that mean?
# Close for me (train only a little higher). So I don't see huge overfitting here, the model isn't memorizing train and failing test on this split.

# Production — keep / drop which features, using the numbers?
# Keep coefs that are clearly not ~0: failures, schoolsup (but we need to interpret carefully), studytime, Medu, Fedu, higher, internet, sex, traveltime.
# First drops for me: activities and freetime (near-zero coefs).

# --- Task 6: Evaluate and summarize ---
os.makedirs("assignments_02/outputs", exist_ok=True)

plt.clf()
plt.scatter(y_pred_f, y_test_f, alpha=0.6, edgecolors="none")
vmin = min(y_pred_f.min(), y_test_f.min())
vmax = max(y_pred_f.max(), y_test_f.max())
plt.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1)
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")

# Does the model seem to struggle more at the high end, the low end, or is error roughly uniform across grade levels?
# For my plot the cloud follows the diagonal most of the middle; I don’t see one side much worse than the other. Error feels roughly uniform, maybe noisier at very high grades because there are fewer of those points.

# What does a value above or below the diagonal mean?
# Above the diagonal: actual G3 is higher than predicted. Below: predicted is higher than actual.

n_filtered = len(df_clean)
n_test = len(y_test_f)
idx_pos = int(np.argmax(model.coef_))
idx_neg = int(np.argmin(model.coef_))
name_pos, coef_pos = feature_cols[idx_pos], model.coef_[idx_pos]
name_neg, coef_neg = feature_cols[idx_neg], model.coef_[idx_neg]

# Plain-language summary for Task 6 
# After I drop G3=0, df_clean is my modeling set. The test set is 20% of that, using the same random_state as in training.
# RMSE is in the same units as G3 on a 0–20 scale, so read it as how big a typical mistake is. R² stays modest here, meaning most of the variation in grades is still coming from things we did not model.
# The largest positive and largest negative coefficients show which features move predicted grades up or down the most when the rest do not change.
# Schoolsup is still the strongest negative. That still feels like the Task 3 story: extra support marks kids who were already struggling, not that tutoring ruins grades.

print("\nTask 6 plain-language summary:")
print("Rows after G3 filter:", n_filtered, "Test rows:", n_test)
print("Test RMSE in grade points:", round(rmse_f, 3), "Test R²:", round(r2_test_f, 4))
print("Largest positive coef:", name_pos, round(coef_pos, 3), "Largest negative coef:", name_neg, round(coef_neg, 3))
print("What surprised me: schoolsup is still the strongest negative. Same explanation as in Task 3: support goes to weaker students.")


# --- Neglected feature: the power of G1 ---
feature_cols_g1 = feature_cols + ["G1"]
x_g1 = df_clean[feature_cols_g1].values
x_train_g1, x_test_g1, y_train_g1, y_test_g1 = train_test_split(
    x_g1, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(x_train_g1, y_train_g1)
r2_test_g1 = model_g1.score(x_test_g1, y_test_g1)

print("\nFull model + G1:")
print("Test R² (no G1):", r2_test_f)
print("Test R² (with G1):", r2_test_g1)

# Does a high R² here mean G1 is causing G3?
# Not really. G1 and G3 are both math grades in the same year, so of course they move together. A big R² shows they line up well; it does not prove that the first-period grade single-handedly creates the final grade.

# Is this a useful model for finding students who might struggle?
# By the time you have G1, you already know something real about how the term is going. That makes G3 easier to predict, but it is not a crystal ball from before the course starts. It is more like updating from midterm information to the final.

# What could educators do to intervene early, before G1 exists?
# They would need signals from before any graded work in that class: attendance, behavior, last year’s record, home support, and so on. That is closer to the Task 5 model with no early grades in the inputs. The score will be lower, but the timing is earlier when it actually helps.

