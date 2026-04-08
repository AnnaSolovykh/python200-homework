import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

data = {
    'name':   ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
    'grade':  [85, 72, 90, 68, 95],
    'city':   ['Boston', 'Austin', 'Boston', 'Denver', 'Austin'],
    'passed': [True, True, True, False, True]
}
df = pd.DataFrame(data)

# --- Pandas ---
# --- Pandas Q1 ---
print(f'Num Rows: {len(df)}')

# --- Pandas Q2 ---
filtered_df = df[(df['passed'] == True) & (df['grade'] > 80)]
print('\nStudents who passed and have grade > 80:')
print(filtered_df)

# --- Pandas Q3 ---
grade_curved = df['grade'] + 5
print('\nCurved Grades:')
print(grade_curved)

# --- Pandas Q4 ---
df['name_upper'] = df['name'].str.upper()
print('\nDataFrame with Uppercase Names:')
print(df)

# --- Pandas Q5 ---
city_avg_grade = df.groupby('city')['grade'].mean();
print('\nAverage Grade by City:')
print(city_avg_grade)

# --- Pandas Q6 ---
df['city'] = df['city'].replace('Austin', 'Houston')
print('\nDataFrame with Updated City Names:')
print(df)

# --- Pandas Q7 ---
top_3 = df.sort_values(by='grade', ascending=False).head(3)
print('\nTop 3 Students by Grade:')
print(top_3)

# --- NumPy ---
# --- NumPy Q1---
array = np.array([10, 20, 30, 40, 50])
print(f'\nShape: {array.shape}')
print(f'DType: {array.dtype}')
print(f'Ndim: {array.ndim}')

# --- NumPy Q2---
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f'\nShape: {arr.shape}')
print(f'Size: {arr.size}')

# --- NumPy Q3---
top_left_2d = arr[0:2, 0:2]
print('\nTop-left 2D Subarray:')
print(top_left_2d)

# --- NumPy Q4---
zeros_array = np.zeros((3, 4))
print('\n3x4 Array of Zeros:')
print(zeros_array)

ones_array = np.ones((2, 5))
print('\n2x5 Array of Ones:')
print(ones_array)

# --- NumPy Q5---
arr5 = np.arange(0, 50, 5)
print('\nArray of Multiples of 5 from 0 to 45:')
print(arr5)
print(f'Shape: {arr5.shape}')
print(f'Mean: {arr5.mean()}')
print(f'Sum: {arr5.sum()}')
print(f'Standard Deviation: {arr5.std()}')

# --- NumPy Q6---
random_values = np.random.normal(0, 1, 200)
print('\nRandom Values from Standard Normal Distribution:')
print(random_values)
print(f'Mean: {random_values.mean()}')
print(f'Standard Deviation: {random_values.std()}')


# --- Matplotlib ---
# --- Matplotlib Q1---
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y)
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.grid() 
plt.show()

# --- Matplotlib Q2---
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.title('Scores by Subject')
plt.xlabel('Subject')
plt.ylabel('Score')
plt.show()

# --- Matplotlib Q3---
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color='purple',label='Dataset 1')
plt.scatter(x2, y2, color='green',label='Dataset 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# --- Matplotlib Q4---
fig, ax = plt.subplots(1, 2)

# Left subplot: line plot (from Q1)
ax[0].plot([0, 1, 2, 3, 4, 5], [0, 1, 4, 9, 16, 25])
ax[0].set_title('Squares')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].grid()

# Right subplot: bar plot (from Q2)
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]
ax[1].bar(subjects, scores)
ax[1].set_title('Scores by Subject')
ax[1].set_xlabel('Subject')
ax[1].set_ylabel('Score')

plt.tight_layout()
plt.show()

# --- Descriptive Statistics Review ---
# --- Descriptive Statistics Q1 ---
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
data_array = np.array(data)

mean = np.mean(data_array)
median = np.median(data_array)
variance = np.var(data_array)
std = np.std(data_array)

print(f'\nData: {data}')
print(f'Mean: {mean}')
print(f'Median: {median}')
print(f'Variance: {variance}')
print(f'Standard Deviation: {std}')

# --- Descriptive Statistics Q2 ---
scores = np.random.normal(65, 10, 500)
plt.hist(scores, bins=30)
plt.title('Distribution of Scores')
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.show()  

# --- Descriptive Statistics Q3 ---
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a, group_b], labels=['Group A', 'Group B'])
plt.title('Comparison of Group A and Group B')
plt.show()

# --- Descriptive Statistics Q4 ---
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.boxplot([normal_data, skewed_data], labels=['Normal', 'Exponential'])
plt.title('Comparison of Normal and Skewed Data')
plt.show()
# Which distribution is more skewed?
# The Exponential distribution looks skewed as it has a long tail with potential outliers. 

# Which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution?  
# For the Normal distribution, the mean would be a more appropriate measure because the data is symmteric. 
# For the Exponential distribution, the median would be a more appropriate measure as this way we will rule out the effect of outliers.

# --- Descriptive Statistics Q5 ---
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

s1 = pd.Series(data1)
s2 = pd.Series(data2)

print(f'\nMean of s1: {s1.mean()}')
print(f'Median of s1: {s1.median()}')
print(f'Mode of s1: {s1.mode()[0]}')
print(f'\nMean of s2: {s2.mean()}')
print(f'Median of s2: {s2.median()}')
print(f'Mode of s2: {s2.mode()[0]}')

# Why are the median and mean so different for data2?
# This happens because the mean is affected by the outlier.

# --- Hypothesis Testing Review ---
# --- Hypothesis Testing Q1 ---
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f'\nT-statistic: {t_stat}')
print(f'P-value: {p_value}')

# --- Hypothesis Testing Q2 ---
if p_value < 0.05:
    print('\nReject the null hypothesis')
else:
    print('Fail to reject the null hypothesis')

# --- Hypothesis Testing Q3 ---
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_value = stats.ttest_rel(before, after)
print(f'\nT-statistic: {t_stat}')
print(f'P-value: {p_value}')

if p_value < 0.05:
    print('\nReject the null hypothesis')
else:
    print('Fail to reject the null hypothesis')

# --- Hypothesis Testing Q4 ---
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_value = stats.ttest_1samp(scores, 70)

print(f'\nT-statistic: {t_stat}')
print(f'P-value: {p_value}')

if p_value < 0.05:
    print('\nReject the null hypothesis')
else:
    print('Fail to reject the null hypothesis')

# --- Hypothesis Testing Q5 ---
t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative='less')
print(f'\nOne-tailed T-statistic: {t_stat}')
print(f'One-tailed P-value: {p_value}')

if p_value < 0.05:
    print('\nReject the null hypothesis')
else:
    print('Fail to reject the null hypothesis')


# --- Hypothesis Testing Q6 ---
mean_a = sum(group_a) / len(group_a)
mean_b = sum(group_b) / len(group_b)

print(f"\nConclusion for Q1: The scores in Group A (mean = {mean_a}) are significantly lower than the scores in Group B (mean = {mean_b}). As the p-value is extremely small, it is unlikely that this difference is due to chance.")

# --- Correlation Review ---
# --- Correlation Q1 ---
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print(f'\nCorrelation Matrix:\n{corr_matrix}')
print(f'Correlation Coefficient: {corr_matrix[0, 1]}')

# I expect a correlation of 1.0 because y = 2*x for every pair. There is a perfect positive connection between them.

# --- Correlation Q2 ---
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
corr_coeff, p_value = pearsonr(x, y)
print(f"\nPearson correlation: {corr_coeff}")
print(f"P-value: {p_value}")

# --- Correlation Q3 ---
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(df.corr())

# --- Correlation Q4 ---
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# --- Correlation Q5 ---
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines ---
# --- Pipelines Q1 ---
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

def data_pipeline(arr):
    step_1 = create_series(arr)
    step_2 = clean_data(step_1)
    step_3 = summarize_data(step_2)
    return step_3

summary_result = data_pipeline(arr)
print("\nPipeline Summary:")
for key, value in summary_result.items():
    print(f"{key}: {value}")