import numpy as np
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- scikit-learn ---
# --- scikit-learn Q1 ---
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)
predictions = model.predict([[4], [8]])

print(f"Predicted salary for year 4: {predictions[0]}")
print(f"Predicted salary for year 8: {predictions[1]}")

# --- scikit-learn Q2 ---
x = np.array([10, 20, 30, 40, 50])
print(x.shape)
x_2d = x.reshape(-1, 1)
print("\nReshaped x: ", x_2d.shape)

# Why does scikit-learn need X to be 2D?

# It needs X to be 2D because it expects the input to be a matrix where each row is a sample and each column is a feature.

# --- scikit-learn Q3 ---
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print("\nCluster centers:", kmeans.cluster_centers_)
print("Cluster labels:")
print(np.bincount(labels))


plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, alpha=0.7)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='black', s=100)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("assignments_02/outputs/kmeans_clusters.png")

# --- Linear Regression ---
# --- Linear Regression Q1 ---
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

plt.clf()
plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.7)

plt.title("\nAge vs Cost")
plt.xlabel("Age")
plt.ylabel("Cost")

plt.savefig("assignments_02/outputs/age_vs_cost.png")

# Are there two distinct groups visible? 
# What does that suggest about the smoker variable?
# Yes, there are two distinct groups visible. The upper group with red dots are presumably smokers. 
# It suggests that smoking status adds significantly to medical costs at any age.

# --- Linear Regression Q2 ---
x = age.reshape(-1, 1)
y = cost
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", x_train.shape)
print("Test set shape:", x_test.shape)
print("Training labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)

# --- Linear Regression Q3 ---
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

print("\nSlope:", lin_reg.coef_[0])
print("Intercept:", lin_reg.intercept_)

y_pred = lin_reg.predict(x_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = lin_reg.score(x_test, y_test)
print("RMSE:", rmse)
print("R-squared (R²):", r2)

# Add a comment interpreting the slope in plain English -- what does it mean for medical costs?

# For each additional year of age, medical costs increase by ~$197.
# The low R² (~0.07) suggests that age alone is a poor predictor of medical costs.

# --- Linear Regression Q4 ---
x_full = np.column_stack((age, smoker))
x_full_train, x_full_test, y_train, y_test = train_test_split(x_full, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(x_full_train, y_train)

print("\nAge coefficient:", model_full.coef_[0])
print("Smoker coefficient:", model_full.coef_[1])

r2_full = model_full.score(x_full_test, y_test)
print("R² (age only):", r2)
print("R² (age + smoker):", r2_full)

# Add a comment interpreting the smoker coefficient: what does it represent in practical terms?

# Being a smoker adds $14,538 to annual medical costs,
# regardless of age. Adding this feature dramatically improved R² from 0.07 to 0.77.
# So adding the smoker flag turned a nearly useless model into a strong one.

# --- Linear Regression Q5 ---
y_pred_full = model_full.predict(x_full_test)

plt.clf()
plt.scatter(y_pred_full, y_test, alpha=0.7)

min_val = min(y_pred_full.min(), y_test.min())
max_val = max(y_pred_full.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted cost")
plt.ylabel("Actual cost")
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")

# Add a comment: what does it mean when a point falls above the diagonal? What about below?

# In a predicted vs actual plot, points above the diagonal mean the model underestimated the true value
# (actual cost is higher than predicted). Points below the diagonal mean the model overestimated it
# (actual cost is lower than predicted). The closer points are to the diagonal, the more accurate the model is.