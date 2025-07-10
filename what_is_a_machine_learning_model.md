
# Machine Learning Models Explained

## Example Dataset

```
X - 1   2   3   4   5
Y - 5   7   9  11  13
```

- **X** is the Feature (Independent Variable)  
- **Y** is the Outcome (Dependent Variable)

If we plot this data on a graph, we get a **linear line**. It is clear that as **X increases, Y also increases**.

## General Equation of a Straight Line

\[ Y = mx + c \]

Where:

- **m** is the slope
- **c** is the intercept

### Example Calculation

Given two points: **p1(2,7)** and **p2(3,9)**

- **Slope (m)**

  \[
  m = (y2 - y)}/(x2 - x1)
  = (9 - 7)/(3 - 2)
  = 2
  \]

- Using this slope to calculate **intercept (c)**:

  We know the target is 13 when the feature is 5:

  \[
  13 = 2(5) + c
  \]

  \[
  c = 3
  \]
Therefore, for this line:

- **Slope (m)** = 2  
- **Intercept (c)** = 3

### What is Slope?

For every **unit change in X**, there is a **2 unit change in Y**.

### What is Intercept?

The **distance of the line from the origin**. In this case, it is **3**.

## Implementing in Code (Example)

```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

➡️ We fit the `x_axis` and `y_axis` values to establish their relationship, enabling predictions for new data based on this relation.

## Definition of a Machine Learning Model

> A **Machine Learning Model** is a **function that finds a relationship** between the features and the target variable.  
> It **learns patterns from data** and makes new predictions based on this learning.

### Important Notes:

- In our example, the relationship is **linear**, but in real scenarios, relationships can be **non-linear** (polynomial, logistic, SVM, K-means clustering).

## Supervised Learning

- **Learns from labeled data**.

### What is a Labeled Dataset?

A dataset where **each data point has input data and an associated correct output (label)**.

### Types of Supervised Learning Algorithms

#### 1. Classification

**Answers:** What type is this?  
Examples: Male/Female, Spam/Ham

**Classification Models:**

- Logistic Regression
- SVM (Support Vector Machine)
- Decision Tree
- K-Nearest Neighbour
- Random Forest
- Naive Bayes Classification

#### 2. Regression

**Answers:** How much? or How many?  
Examples: Temperature, House price

**Regression Models:**

- Linear Regression
- Lasso Regression
- Polynomial Regression
- SVM
- Random Forest
- Bayesian Linear Regressor

## Unsupervised Learning

- **Learns patterns without labeled outputs**.
- Discovers structure in data **on its own**.

### Tasks in Unsupervised Learning

| Task                    | Description                        | Example                        |
|--------------------------|-------------------------------------|--------------------------------|
| Clustering               | Group similar data points          | Segmenting customers          |
| Dimensionality Reduction | Reduce number of features          | Visualizing high-dimensional data |
| Association              | Find rules/patterns between features | "People who buy X also buy Y" |

### Common Unsupervised Algorithms

- K-Means
- Hierarchical Clustering
- PCA (Principal Component Analysis)
- Apriori / Eclat

## When to Choose Which Model?

### 1. Based on Data Type

- **Images and Videos** → CNN (Convolutional Neural Networks)
- **Text and Speech Data** → RNN (Recurrent Neural Networks)
- **Numerical Data** → SVM, Logistic Regression, Decision Trees

### 2. Based on Task

- **Classification Tasks (this or that?)**
  - SVM, Logistic Regression, Decision Trees, etc.
- **Regression Tasks (how much?)**
  - Linear Regression, Random Forest, Polynomial Regression
- **Clustering Tasks (grouping)**
  - K-Means Clustering, Hierarchical Clustering
