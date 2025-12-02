# Assignment 1: Perform the following operations using Python on a data set : read data
# from different formats(like csv, xls),indexing and selecting data, sort data,
# describe attributes of data, checking data types of each column. (Use
# Titanic Dataset).
import pandas as pd
# Read Data
csvdf=pd.read_csv("titanic.csv")
exceldf=pd.read_excel("titanic.xlsx")
# Indexing and Selecting
csvdf.iloc[0:4]
csvdf.iloc[-2:]
csvdf['Name']
csvdf[['Name',"Pclass"]]
exceldf[["Name","Pclass"]]
# sort data
csvdf.sort_values(by="Age",ascending=False)
exceldf.sort_values(by="Name")
# describe attributes of data
csvdf.describe()
exceldf.describe()
# data types
csvdf.dtypes

# --------------------------------------------------------------------
#2: Perform the following operations using Python on the Telecom_Churn
# dataset. Compute and display summary statistics for each feature available
# in the dataset using separate commands for each statistic. (e.g. minimum
# value, maximum value, mean, range, standard deviation, variance and
# percentiles).

import pandas as pd
df=pd.read_csv("telecom_churn.csv")
# minimum
df.min(numeric_only=True)
# maximum
df.max(numeric_only=True)
# mean
df.mean(numeric_only=True)
# range
df.max(numeric_only=True)-df.min(numeric_only=True)
# standard deviation
df.std(numeric_only=True)
# variance
df.var(numeric_only=True)
# percentiles
df.quantile([0.25,0.50,0.75],numeric_only=True)

# -------------------------------------------------------------------------
# 3:Perform the following operations using Python on the data set
# House_Price Prediction dataset. Compute standard deviation, variance and
# percentiles using separate commands, for each feature. Create a histogram
# for each feature in the dataset to illustrate the feature distributions.
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("house_price.csv")
# standard deviation
df.std(numeric_only=True)
# variance
df.var(numeric_only=True)
# percentiles
df.quantile([.25,.50,.75],numeric_only=True)
# histogram for each feature
df.hist(figsize=(10,12))
plt.show()

# ----------------------------------------------------------------------
# 4. Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Implement step by step
# using commands - Dont use library) Use this dataset to build a decision
# tree, with Buys as the target variable, to help in buying lipsticks in the
# future. Find the root node of the decision tree.

import csv
import math

# Read CSV file
data = []
with open("cosmetics.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data.append(row)

# target column
target = header.index("Buys")

# Calculate entropy
def entropy(rows):
    total = len(rows)
    yes = sum(1 for r in rows if r[target] == "Yes")
    no = total - yes
    if yes == 0 or no == 0:
        return 0
    p_yes = yes / total
    p_no = no / total
    return -(p_yes*math.log2(p_yes) + p_no*math.log2(p_no))

# Information gain
def info_gain(rows, col):
    base_entropy = entropy(rows)
    values = {}
    for r in rows:
        values.setdefault(r[col], []).append(r)

    remainder = 0
    for v in values.values():
        remainder += (len(v)/len(rows)) * entropy(v)

    return base_entropy - remainder

# Compute IG for all attributes except Buys
best_attr = None
best_ig = -1

for i, col in enumerate(header):
    if col == "Buys": 
        continue
    ig = info_gain(data, i)
    print(col, "-> IG =", round(ig, 4))
    if ig > best_ig:
        best_ig = ig
        best_attr = col

print("\nRoot Node =", best_attr)

# -------------------------------------------------------------------------
# 5,6,7,8: Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Use library commands)
# According to the decision tree you have made from the previous training
# data set, what is the decision for the test data: [Age &lt; 21, Income = Low,
# Gender = Female, Marital Status = Married]?
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv("cosmetics.csv")
# Encode the Data
encoder={}
for col in ["Income","Gender","MaritalStatus","Buys"]:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    encoder[col]=le

# Tree
model=DecisionTreeClassifier()
x=df.drop("Buys",axis=1)
y=df["Buys"]
model.fit(x,y)

# Test Data
test=pd.DataFrame({
    "Age":[21],
    "Income":["Low"],
    "Gender":["Female"],
    "MaritalStatus":["Married"]
})

# Encode test data
for col in ["Income","Gender","MaritalStatus"]:
    test[col]=encoder[col].transform(test[col])

result=model.predict(test)[0]
if result==1:
    print("YES")
else:
    print("NO")

# --------------------------------------------------------------------
# 9. Write a program to do the following: You have given a collection of 8
# points. P1=[0.1,0.6] P2=[0.15,0.71] P3=[0.08,0.9] P4=[0.16, 0.85]
# P5=[0.2,0.3] P6=[0.25,0.5] P7=[0.24,0.1] P8=[0.3,0.2]. Perform the k-mean
# clustering with initial centroids as m1=P1 =Cluster#1=C1 and
# m2=P8=cluster#2=C2. Answer the following 1] Which cluster does P6
# belong to? 2] What is the population of a cluster around m2? 3] What is
# the updated value of m1 and m2?

import math
points={
    "P1":[0.1,0.6],
    "P2":[0.15,0.71],
    "P3":[0.08,0.9],
    "P4":[0.16, 0.85],
    "P5":[0.2,0.3],
    "P6":[0.25,0.5],
    "P7":[0.24,0.1],
    "P8":[0.3,0.2]
}
m1=points["P1"]
m2=points["P8"]

c1=[]
c2=[]

def dist(a,b):
    return math.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2)

for name,point in points.items():
    d1=dist(m1,point)
    d2=dist(m2,point)
    if d1<d2:
        c1.append(name)
    else:
        c2.append(name)

print("C1", c1)
print("C2", c2)

for c in c1:
    if c=="P6":
        print("P6 in C1")

for c in c2:
    if c=="P6":
        print("P6 in C2")

print("Population of a cluster around m2 is:",len(c2))   

x1=sum(points[c][0] for c in c1)/len(c1)
x2=sum(points[c][1] for c in c1)/len(c1)

y1=sum(points[c][0] for c in c2)/len(c2)
y2=sum(points[c][1] for c in c2)/len(c2)

print("Updated value of m1 is:",[x1,x2])
print("Updated value of m2 is:",[y1,y2])    

# --------------------------------------------------------------------
# 10. Write a program to do the following: You have given a collection of 8
# points. P1=[2, 10] P2=[2, 5] P3=[8, 4] P4=[5, 8] P5=[7,5] P6=[6, 4] P7=[1, 2]
# P8=[4, 9]. Perform the k-mean clustering with initial centroids as m1=P1
# =Cluster#1=C1 and m2=P4=cluster#2=C2, m3=P7 =Cluster#3=C3. Answer
# the following 1] Which cluster does P6 belong to? 2] What is the
# population of a cluster around m3? 3] What is the updated value of m1,
# m2, m3?

import math
points={
    "P1":[2, 10],
    "P2":[2, 5],
    "P3":[8,4],
    "P4":[5,8],
    "P5":[7,5],
    "P6":[6,4],
    "P7":[1,2],
    "P8":[4,9],
}
m1=points["P1"]
m2=points["P4"]
m3=points["P7"]

c1=[]
c2=[]
c3=[]

def dist(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

for name, point in points.items():
    d1=dist(point,m1)
    d2=dist(point,m2)
    d3=dist(point,m3)
    d_min=min(d1,d2,d3)

    if d_min==d1:
        c1.append(name)
    elif d_min==d2:
        c2.append(name)
    else:
        c3.append(name)


print("C1:",c1)
print("C2:",c2)
print("C3:",c3)

#  Which cluster does P6 belong to
for c in c1:
    if c=="P6":
        print("P6 present in C1")

for c in c2:
    if c=="P6":
        print("P6 present in C2")   

for c in c3:
    if c=="P6":
        print("P6 present in C3")


print("Population of C3:",len(c3))

x1=sum(points[c][0] for c in c1)/len(c1)
x2=sum(points[c][1] for c in c1)/len(c1)

y1=sum(points[c][0] for c in c2)/len(c2)
y2=sum(points[c][1] for c in c2)/len(c2)

z1=sum(points[c][0] for c in c3)/len(c3)
z2=sum(points[c][1] for c in c3)/len(c3)

print("Update M1 :",[x1,x2])
print("Update M2 :",[y1,y2])
print("Update M3 :",[z1,z2])

# ---------------------------------------------------------------------
# 11. Use Iris flower dataset and perform following :
# 1. List down the features and their types (e.g., numeric, nominal)
# available in the dataset. 2. Create a histogram for each feature in the
# dataset to illustrate the feature distributions.

import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("iris")

print(df.dtypes)
df.hist(figsize=(10,12))
plt.suptitle("Iris Feature Distributions")
plt.show()

# -------------------------------------------------------------------
# 12.Use Iris flower dataset and perform following :
# 1. Create a box plot for each feature in the dataset.
# 2. Identify and discuss distributions and identify outliers from them.

import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("iris")

df.boxplot(figsize=(8,10))
plt.suptitle("Box Plot for Iris Dataset Features")
plt.show()
df.describe()

# -----------------------------------------------------------------------
# 13. Use the covid_vaccine_statewise.csv dataset and perform the
# following analytics.
# a. Describe the dataset
# b. Number of persons state wise vaccinated for first dose in India
# c. Number of persons state wise vaccinated for second dose in India

import pandas as pd
df=pd.read_csv("covid_vaccine_statewise.csv")
# df.describe()
# df[["State","First Dose Administered"]]
# df[["State","Second Dose Administered"]]

df["Updated On"]=pd.to_datetime(df["Updated On"])
df=df.sort_values(["State","Updated On"])
latest=df.groupby("State").tail(1)

print(df.describe(include="all"))
first_dose = latest[["State", "First Dose Administered"]]
print(first_dose)
second_dose = latest[["State", "Second Dose Administered"]]
print(second_dose)

# -----------------------------------------------------------------------------------
# 14. Use the covid_vaccine_statewise.csv dataset and perform the
# following analytics.
# A. Describe the dataset.
# B. Number of Males vaccinated
# C.. Number of females vaccinated

import pandas as pd
df=pd.read_csv("covid_vaccine_statewise.csv")
print(df.describe())

print(df[["State","Male Vaccinated"]])
total=0
for n in df['Male Vaccinated']:
    total+=n
print("Total Male Vaccinated:",total)


print(df[["State","Female Vaccinated"]])
total=0
for n in df['Female Vaccinated']:
    total+=n
print("Total Male Vaccinated:",total)

# -------------------------------------------------------------------
# 15. Use the dataset 'titanic'. The dataset contains 891 rows and contains
# information about the passengers who boarded the unfortunate Titanic
# ship. Use the Seaborn library to see if we can find any patterns in the data.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic=sns.load_dataset("titanic")
titanic.isnull().sum()
sns.countplot(x=titanic["survived"])
plt.show()
sns.countplot(x=titanic["sex"],hue=titanic["survived"])
plt.show()
sns.countplot(x=titanic["pclass"],hue=titanic["survived"])
plt.show()
sns.histplot(x=titanic["age"],hue=titanic["survived"],kde=True)
plt.show()
sns.boxplot(x="survived", y="fare", data=titanic)
plt.show()
# Women survived more than men
# 1st class passengers survived more than 2nd and 3rd class
# Children (age < 15) had higher survival rate
# High-fare passengers survived more
# Passengers with small families survived more
# People who embarked from Cherbourg survived more
# Most deaths were in adult males from 3rd class

# -----------------------------------------------------------------------
# 16. Use the inbuilt dataset &#39;titanic&#39;. The dataset contains 891 rows and
# contains information about the passengers who boarded the unfortunate
# Titanic ship. Write a code to check how the price of the ticket (column
# name:fare) for each passenger is distributed by plotting a histogram.

import seaborn as sns
import matplotlib.pyplot as plt
titanic=sns.load_dataset("titanic")

sns.histplot(x=titanic["fare"],bins=30,kde=True)
plt.title("Distribution of Ticket Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# ------------------------------------------------------------------------
# 17. Compute Accuracy, Error rate, Precision, Recall for following confusion
# matrix ( Use formula for each)

# Confusion Matrix Metrics (Compact One-Page Answer)
# Given: TP=1, FP=1, FN=8, TN=90, Total=100
# Accuracy = (TP+TN)/Total = (1+90)/100 = 0.91 = 91%
# Meaning: Model correct 91% times.
# Error Rate = (FP+FN)/Total = (1+8)/100 = 0.09 = 9%
# Meaning: Model wrong 9% times.
# Precision = TP/(TP+FP) = 1/(1+1) = 0.5 = 50%
# Meaning: Positive predictions are 50% correct.
# Recall = TP/(TP+FN) = 1/(1+8) ≈ 0.11 = 11%
# Meaning: Model finds only 11% actual positives (misses many).

# -------------------------------------------------------------------------------
# 18. Use House_Price prediction dataset. Provide summary statistics (mean,
# median, minimum, maximum, standard deviation) of variables (categorical
# vs quantitative) such as- For example, if categorical variable is age groups
# and quantitative variable is income, then provide summary statistics of
# income grouped by the age groups.
import pandas as pd
data = {
    "Location": ["City", "City", "Village", "Town", "Town", "Village", "City", "Town", "Village"],
    "House_Price": [7500000, 8200000, 3500000, 5500000, 6000000, 4000000, 9000000, 5800000, 3800000]
}
df=pd.DataFrame(data)

summury=df.groupby("Location")["House_Price"].agg(["mean","median","min","max","std"])
print(summury)

# ------------------------------------------------------------------------------------
# 19. Write a Python program to display some basic statistical details like
# percentile, mean, standard deviation etc (Use python and pandas
# commands) the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’
# of iris.csv dataset.

import seaborn as sns
df=sns.load_dataset('iris')
species_list=["versicolor","setosa","virginica"]

for specie in species_list:
    data=df[df["species"]==specie]
    print(data.describe())

# -------------------------------------------------------------------------------------
# 22. Compute Accuracy, Error rate, Precision, Recall for the following
# confusion matrix.
# TP = 90
# FN = 210
# FP = 140
# TN = 9560
# Total = 10000
# Formulas & Answers
# Accuracy = (TP + TN) / Total = (90 + 9560) / 10000 = 96.5%
# Error Rate = (FP + FN) / Total = (140 + 210) / 10000 = 3.5%
# Precision = TP / (TP + FP) = 90 / 230 = 39.1%
# Recall = TP / (TP + FN) = 90 / 300 = 30%

# ---------------------------------------------------------------------------------
# 23. With reference to Table , obtain the Frequency table for the
# attribute age. From the frequency table you have obtained, calculate
# the information gain of the frequency table while splitting on Age. (Use
# step by step Python/Pandas commands)
import pandas as pd
import math

# STEP 1: Create DataFrame
data = {
    "Age": ["Young","Young","Middle","Old","Old","Old","Middle","Young",
            "Young","Old","Young","Middle","Middle","Old"],
    "Income": ["High","High","High","Medium","Low","Low","Low","Medium",
               "Low","Medium","Medium","Medium","High","Medium"],
    "Married": ["No","No","No","No","Yes","Yes","No","No",
                "Yes","Yes","No","No","Yes","No"],
    "Health": ["Fair","Good","Fair","Fair","Fair","Good","Good","Fair",
               "Fair","Fair","Fair","Good","Fair","Good"],
    "Class": ["No","No","Yes","Yes","Yes","No","Yes","No",
              "Yes","Yes","Yes","Yes","Yes","No"]
}
df=pd.DataFrame(data)

# STEP 2: Frequency Table for Age
freq=df["Age"].value_counts()
print(freq)

# STEP 3: Entropy Function
def entropy(column):
    values=column.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in values)

# STEP 4: Total Entropy of dataset
overall_entropy=entropy(df["Class"])
print(overall_entropy)

# STEP 5: Entropy of each Age group
entropy_age_groups=df.groupby("Age")["Class"].apply(entropy)
print(entropy_age_groups)

# STEP 6: Weighted Entropy after split on Age
total_rows=len(df)
weighted_entropy = sum(
    (df[df["Age"]==age].shape[0]/total_rows)*entropy_age_groups[age]
    for age in entropy_age_groups.index
)
print(weighted_entropy)

# STEP 7: Information Gain
information_gain=overall_entropy-weighted_entropy
print("\nInformation Gain (Age):",information_gain)

# ---------------------------------------------------------------------------
# 24. Perform the following operations using Python on a suitable data set,
# counting unique values of data, format of each column, converting
# variable data type (e.g. from long to short, vice versa), identifying missing
# values and filling in the missing values.

import pandas as pd
df=pd.read_csv("students.csv")

# Count unique values of each column
print("Unique values:")
print(df.nunique())

# format of each column
print("\nColumn Data Types:")
print(df.dtypes)

# converting variable data type (e.g. from long to short, vice versa)
df["Age"]=df["Age"].astype("float")
df["Passed"]=df["Passed"].astype("category")
print("\nData Types After conversion")
print(df.dtypes)

# identifying missing values 
print("\nMissing Values:")
print(df.isnull().sum())

# Fill the Missing the values
df["Marks"].fillna(df["Marks"].min(),inplace=True)
df["Height"].fillna(df["Height"].median(),inplace=True)
print("\nAfter Filling Missing Values:")
df
# -----------------------------------------------------------------------------
# 25. Perform Data Cleaning, Data transformation using Python on any data
# set.

import pandas as pd
df = pd.read_csv("data_cleaning.csv")
print(df)
# 2. Check format (data types)
print("\nData Types:")
print(df.dtypes)
# 3. Count unique values
print("\nUnique values in each column:")
print(df.nunique())
# 4. Identify missing values
print("\nMissing Values:")
print(df.isnull().sum())
# 5. Fill missing Age with mean
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Salary"]=df["Salary"].fillna(df["Salary"].median())
df["City"]=df["City"].fillna("Unknown")
# 6. Convert Age from float to int
df["Age"]=df["Age"].astype(int)
df["Salary"]=df["Salary"].astype(int)
# 7. Transformation
df["Name"]=df["Name"].str.upper()
# 8. Print Clean data
print("\nCleaned & Transformed Dataset:")
df
# ---------------------------------------------------------------------------------
# 4. Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Implement step by step
# using commands - Dont use library) Use this dataset to build a decision
# tree, with Buys as the target variable, to help in buying lipsticks in the
# future. Find the root node of the decision tree.

import csv
import math

# Read CSV file
data = []
with open("cosmetics.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data.append(row)

# target column
target = header.index("Buys")

# Calculate entropy
def entropy(rows):
    total = len(rows)
    yes = sum(1 for r in rows if r[target] == "Yes")
    no = total - yes
    if yes == 0 or no == 0:
        return 0
    p_yes = yes / total
    p_no = no / total
    return -(p_yes*math.log2(p_yes) + p_no*math.log2(p_no))

# Information gain
def info_gain(rows, col):
    base_entropy = entropy(rows)
    values = {}
    for r in rows:
        values.setdefault(r[col], []).append(r)

    remainder = 0
    for v in values.values():
        remainder += (len(v)/len(rows)) * entropy(v)

    return base_entropy - remainder

# Compute IG for all attributes except Buys
best_attr = None
best_ig = -1

for i, col in enumerate(header):
    if col == "Buys": 
        continue
    ig = info_gain(data, i)
    print(col, "-> IG =", round(ig, 4))
    if ig > best_ig:
        best_ig = ig
        best_attr = col

print("\nRoot Node =", best_attr)
# ---------------------------------------------------------------------------------
# 20:Write a program to cluster a set of points using K-means for IRIS
# dataset. Consider, K=3, clusters. Consider Euclidean distance as the
# distance measure. Randomly initialize a cluster mean as one of the data
# points. Iterate at least for 10 iterations. After iterations are over, print the
# final cluster means for each of the clusters.
# COMMAND 1: Import required libraries
import seaborn as sns    # If this fails, use: import seaborn as sns
import numpy as np
import random

# COMMAND 2: Load IRIS dataset
iris = sns.load_dataset("iris")
print("First 5 rows of IRIS dataset:")
print(iris.head())

# COMMAND 3: Select numeric columns for clustering
data = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
print("\nData shape:", data.shape)

# COMMAND 4: Set number of clusters K = 3
K = 3

# COMMAND 5: Randomly choose 3 initial cluster means from data points
np.random.seed(4)
initial_indices = np.random.choice(len(data), K, replace=False)
means = data[initial_indices]

print("\nInitial Random Cluster Means:")
print(means)

# COMMAND 6: Function to compute Euclidean distance
def euclidean(p, q):
    return np.sqrt(np.sum((p - q)**2))

# COMMAND 7: K-means iteration (run for 10 iterations)
for iteration in range(10):
    print(f"\n--- ITERATION {iteration+1} ---")
    
    # Step 1: Assign each point to nearest cluster
    clusters = {i: [] for i in range(K)}

    for point in data:
        # compute distance from point to each mean
        distances = [euclidean(point, mean) for mean in means]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    # Step 2: Recalculate new means
    new_means = []
    for i in range(K):
        if len(clusters[i]) > 0:
            new_mean = np.mean(clusters[i], axis=0)
        else:
            new_mean = means[i]   # if empty cluster
        new_means.append(new_mean)

    # Convert list to numpy array
    new_means = np.array(new_means)

    print("Updated Cluster Means:")
    print(new_means)

    # Update means
    means = new_means

# COMMAND 8: Print final means after 10 iterations
print("\n==============================")
print("FINAL CLUSTER MEANS (AFTER 10 ITERATIONS--------)")
print("==============================")
for i, mean in enumerate(means):
    print(f"Cluster {i+1} mean =", mean)

# ------------------------------------------------------------------------------
# 21:Write a program to cluster a set of points using K-means for IRIS
# dataset. Consider, K=4, clusters. Consider Euclidean distance as the
# distance measure. Randomly initialize a cluster mean as one of the data
# points. Iterate at least for 10 iterations. After iterations are over, print the
# final cluster means for each of the clusters.
# COMMAND 1: Import required libraries
import seaborn as sns
import numpy as np

# COMMAND 2: Load the IRIS dataset
iris = sns.load_dataset("iris")
print("First 5 rows:")
print(iris.head())

# COMMAND 3: Extract only numeric columns for K-means
data = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
print("\nData Shape =", data.shape)

# COMMAND 4: Set number of clusters K = 4
K = 4

# COMMAND 5: Randomly initialize cluster means from dataset rows
np.random.seed(10)  # For reproducibility
initial_indices = np.random.choice(len(data), K, replace=False)
means = data[initial_indices]

print("\nInitial Random Cluster Means:")
print(means)

# COMMAND 6: Euclidean Distance Function
def euclidean(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

# COMMAND 7: Run K-Means for 10 iterations
for iteration in range(10):
    print(f"\n--- ITERATION {iteration + 1} ---")
    
    # Assign points to nearest mean
    clusters = {i: [] for i in range(K)}
    for point in data:
        distances = [euclidean(point, mean) for mean in means]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    # Compute new means
    new_means = []
    for i in range(K):
        if len(clusters[i]) > 0:
            new_means.append(np.mean(clusters[i], axis=0))
        else:
            new_means.append(means[i])  # keep old mean if cluster empty

    new_means = np.array(new_means)
    print("Updated Means:")
    print(new_means)

    # Update means
    means = new_means

# COMMAND 8: Print final cluster means
print("\n==============================")
print("FINAL CLUSTER MEANS (AFTER 10 ITERATIONS)")
print("==============================")
for i, mean in enumerate(means):
    print(f"Cluster {i+1} Mean =", mean)

