# Detect data bias with Amazon SageMaker Clarify

### Introduction


Bias can be present in your data before any model training occurs. Inspecting the dataset for bias can help detect collection gaps, inform your feature engineering, and understand societal biases the dataset may reflect. In this lab I will analyze bias on the dataset, generate and analyze bias report, and prepare the dataset for the model training.<br>
Libraries used: (boto3, sagemaker, pandas, numpy, botocore, matplotlib, seaborn, YPython, sagemaker)<br>
AWS services: (S3, SageMaker)

### Table of Contents

[1. Analyze the dataset](#1-analyze)<br>
[2. Analyze class imbalance on the dataset with Amazon SageMaker Clarify](#2-clarify)<br>
[3. Balance the dataset by "product_category" and "sentiment"](#3-balance)<br>
[4. Analyze bias on balanced dataset with Amazon SageMaker Clarify](#4-analyze)<br>

* Check all the steps in C1_W2_Assignment(1).ipynb
  
<a name='1-analyze'></a>
### 1. Analyze the dataset
Our dataset contains the reviews of a Women's Clothing E-Commerce sales.<br>
Shape, 3 columns, ('sentiment', 'review_body', 'product_category')<br>
-Column: 'sentiment', int type from -1 to 1 (positive, neutral or negative)<br>
-Column: 'review_body', str type with the review of the customer<br>
-Column: 'prudct_category', str with the product category.<br>
* Head<br>
![Screenshot from 2022-10-26 18-33-51](https://user-images.githubusercontent.com/78228205/198084120-3d35eb0a-bdcb-4756-ad59-0529209fae5d.png)<br>

* 'sentiment' by category:<br>
![Screenshot from 2022-10-26 18-43-20](https://user-images.githubusercontent.com/78228205/198085939-cc111240-4b91-4ee0-9d99-bac7863ef7f3.png)<br>
As we can see there are way more positive reviews than negative or neutral. Our a dataset is unbalanced. 

<a name='2-clarify'></a>
### 2. Analyze class imbalance on the dataset with Amazon SageMaker Clarify
Let's analyze bias in sentiment with respect to the product_category facet on the dataset using Clarify with the following metrics:


<details>
  <summary>Class Imbalance (CI)</summary>
  
  ### Class imbalance (CI)
Class imbalance (CI) bias occurs when a facet value d has fewer training samples when compared with another facet a in the dataset. This is because models preferentially fit the larger facets at the expense of the smaller facets and so can result in a higher training error for facet d. Models are also at higher risk of overfitting the smaller data sets, which can cause a larger test error for facet d. Consider the example where a machine learning model is trained primarily on data from middle-aged individuals (facet a), it might be less accurate when making predictions involving younger and older people (facet d).

The formula for the (normalized) facet imbalance measure:

        CI = (na - nd)/(na + nd)

Where na is the number of members of facet a and nd the number for facet d. Its values range over the interval [-1, 1].

    Positive CI values indicate the facet a has more training samples in the dataset and a value of 1 indicates the data only contains members of the facet a.

    Values of CI near zero indicate a more equal distribution of members between facets and a value of zero indicates a perfectly equal partition between facets and represents a balanced distribution of samples in the training data.

    Negative CI values indicate the facet d has more training samples in the dataset and a value of -1 indicates the data only contains members of the facet d.

    CI values near either of the extremes values of -1 or 1 are very imbalanced and are at a substantial risk of making biased predictions.

If a significant facet imbalance is found to exist among the facets, you might want to rebalance the sample before proceeding to train models on it.
</details>

<details>
  <summary>Difference in Positive Proportions in Labels (DPL)</summary>
  
  ### Difference in Positive Proportions in Labels (DPL)
The difference in proportions of labels (DPL) compares the proportion of observed outcomes with positive labels for facet d with the proportion of observed outcomes with positive labels of facet a in a training dataset. For example, you could use it to compare the proportion of middle-aged individuals (facet a) and other age groups (facet d) approved for financial loans. Machine learning models try to mimic the training data decisions as closely as possible. So a machine learning model trained on a dataset with a high DPL is likely to reflect the same imbalance in its future predictions.

The formula for the difference in proportions of labels is as follows:

        DPL = (qa - qd)

Where:

    qa = na(1)/na is the proportion of facet a who have an observed label value of 1. For example, the proportion of a middle-aged demographic who get approved for loans. Here na(1) represents the number of members of facet a who get a positive outcome and na the is number of members of facet a.

    qd = nd(1)/nd is the proportion of facet d who have an observed label value of 1. For example, the proportion of people outside the middle-aged demographic who get approved for loans. Here nd(1) represents the number of members of the facet d who get a positive outcome and nd the is number of members of the facet d.

If DPL is close enough to 0, then we say that demographic parity has been achieved.

For binary and multicategory facet labels, the DPL values range over the interval (-1, 1). For continuous labels, we set a threshold to collapse the labels to binary.

    Positive DPL values indicate that facet a is has a higher proportion of positive outcomes when compared with facet d.

    Values of DPL near zero indicate a more equal proportion of positive outcomes between facets and a value of zero indicates perfect demographic parity.

    Negative DPL values indicate that facet d has a higher proportion of positive outcomes when compared with facet a.

Whether or not a high magnitude of DPL is problematic varies from one situation to another. In a problematic case, a high-magnitude DPL might be a signal of underlying issues in the data. For example, a dataset with high DPL might reflect historical biases or prejudices against age-based demographic groups that would be undesirable for a model to learn.
</details>
<details>
  <summary>Jensen-Shannon Divergence (JS)</summary>
  
  ### Jensen-Shannon Divergence (JS)
The Jensen-Shannon divergence (JS) measures how much the label distributions of different facets diverge from each other entropically. It is based on the Kullback-Leibler divergence, but it is symmetric.

The formula for the Jensen-Shannon divergence is as follows:

        JS = ½*[KL(Pa || P) + KL(Pd || P)]

Where P = ½( Pa + Pd ), the average label distribution across facets a and d.

The range of JS values for binary, multicategory, continuous outcomes is [0, ln(2)).

    Values near zero mean the labels are similarly distributed.

    Positive values mean the label distributions diverge, the more positive the larger the divergence.

This metric indicates whether there is a big divergence in one of the labels across facets. 
</details>

<details>
  <summary>Kullback-Liebler Divergence (KL)</summary>
  
  ### Kullback-Liebler Divergence (KL)
The Kullback-Leibler divergence (KL) measures how much the observed label distribution of facet a, Pa(y), diverges from distribution of facet d, Pd(y). It is also known as the relative entropy of Pa(y) with respect to Pd(y) and quantifies the amount of information lost when moving from Pa(y) to Pd(y).

The formula for the Kullback-Leibler divergence is as follows:

        KL(Pa || Pd) = ∑yPa(y)*log[Pa(y)/Pd(y)]

It is the expectation of the logarithmic difference between the probabilities Pa(y) and Pd(y), where the expectation is weighted by the probabilities Pa(y). This is not a true distance between the distributions as it is asymmetric and does not satisfy the triangle inequality. The implementation uses natural logarithms, giving KL in units of nats. Using different logarithmic bases gives proportional results but in different units. For example, using base 2 gives KL in units of bits.

For example, assume that a group of applicants for loans have a 30% approval rate (facet d) and that the approval rate for other applicants (facet a) is 80%. The Kullback-Leibler formula gives you the label distribution divergence of facet a from facet d as follows:

        KL = 0.8*ln(0.8/0.3) + 0.2*ln(0.2/0.7) = 0.53

There are two terms in the formula here because labels are binary in this example. This measure can be applied to multiple labels in addition to binary ones. For example, in a college admissions scenario, assume an applicant may be assigned one of three category labels: yi = {y0, y1, y2} = {rejected, waitlisted, accepted}.

Range of values for the KL metric for binary, multicategory, and continuous outcomes is [0, +∞).

    Values near zero mean the outcomes are similarly distributed for the different facets.

    Positive values mean the label distributions diverge, the more positive the larger the divergence.

</details>

<details>
  <summary>Kolmogorov-Smirnov Distance (KS)</summary>
  
  ### Kolmogorov-Smirnov Distance (KS)
The Kolmogorov-Smirnov bias metric (KS) is equal to the maximum divergence between labels in the distributions for facets a and d of a dataset. The two-sample KS test implemented by SageMaker Clarify complements the other measures of label imbalance by finding the most imbalanced label.

The formula for the Kolmogorov-Smirnov metric is as follows:

        KS = max(|Pa(y) - Pd(y)|)

For example, assume a group of applicants (facet a) to college are rejected, waitlisted, or accepted at 40%, 40%, 20% respectively and that these rates for other applicants (facet d) are 20%, 10%, 70%. Then the Kolmogorov-Smirnov bias metric value is as follows:

KS = max(|0.4-0.2|, |0.4-0.1|, |0.2-0.7|) = 0.5

This tells us the maximum divergence between facet distributions is 0.5 and occurs in the acceptance rates. There are three terms in the equation because labels are multiclass of cardinality three.

The range of LP values for binary, multicategory, and continuous outcomes is [0, +1], where:

    Values near zero indicate the labels were evenly distributed between facets in all outcome categories. For example, both facets applying for a loan got 50% of the acceptances and 50% of the rejections.

    Values near one indicate the labels for one outcome were all in one facet. For example, facet a got 100% of the acceptances and facet d got none.

    Intermittent values indicate relative degrees of maximum label imbalance.

</details>

<details>
  <summary>L-p Norm (LP)</summary>
  
  ### L-p Norm (LP)
The Lp-norm (LP) measures the p-norm distance between the facet distributions of the observed labels in a training dataset. This metric is non-negative and so cannot detect reverse bias.

The formula for the Lp-norm is as follows:

        Lp(Pa, Pd) = ( ∑y||Pa - Pd||p)1/p

Where the p-norm distance between the points x and y is defined as follows:

        Lp(x, y) = (|x1-y1|p + |x2-y2|p + … +|xn-yn|p)1/p

The 2-norm is the Euclidean norm. Assume you have an outcome distribution with three categories, for example, yi = {y0, y1, y2} = {accepted, waitlisted, rejected} in a college admissions multicategory scenario. You take the sum of the squares of the differences between the outcome counts for facets a and d. The resulting Euclidean distance is calculated as follows:

        L2(Pa, Pd) = [(na(0) - nd(0))2 + (na(1) - nd(1))2 + (na(2) - nd(2))2]1/2

Where:

    na(i) is number of the ith category outcomes in facet a: for example na(0) is number of facet a acceptances.

    nd(i) is number of the ith category outcomes in facet d: for example nd(2) is number of facet d rejections.

    The range of LP values for binary, multicategory, and continuous outcomes is [0, √2), where:

        Values near zero mean the labels are similarly distributed.

        Positive values mean the label distributions diverge, the more positive the larger the divergence.

</details>

<details>
  <summary>Total Variation Distance (TVD)</summary>
  
  ### Total Variation Distance (TVD)
The total variation distance data bias metric (TVD) is half the L1-norm. The TVD is the largest possible difference between the probability distributions for label outcomes of facets a and d. The L1-norm is the Hamming distance, a metric used compare two binary data strings by determining the minimum number of substitutions required to change one string into another. If the strings were to be copies of each other, it determines the number of errors that occurred when copying. In the bias detection context, TVD quantifies how many outcomes in facet a would have to be changed to match the outcomes in facet d.

The formula for the Total variation distance is as follows:

        TVD = ½*L1(Pa, Pd)

For example, assume you have an outcome distribution with three categories, yi = {y0, y1, y2} = {accepted, waitlisted, rejected}, in a college admissions multicategory scenario. You take the differences between the counts of facets a and d for each outcome to calculate TVD. The result is as follows:

        L1(Pa, Pd) = |na(0) - nd(0)| + |na(1) - nd(1)| + |na(2) - nd(2)|

Where:

    na(i) is number of the ith category outcomes in facet a: for example na(0) is number of facet a acceptances.

    nd(i) is number of the ith category outcomes in facet d: for example nd(2) is number of facet d rejections.

    The range of TVD values for binary, multicategory, and continuous outcomes is [0, 1), where:

        Values near zero mean the labels are similarly distributed.

        Positive values mean the label distributions diverge, the more positive the larger the divergence.

</details>

Once processed we obtain a report with values for all categories using the selected metrics<br>
* Report category proportion from 'category_product' unbalanced dataset:<br>
  ![Screenshot from 2022-10-26 19-21-55](https://user-images.githubusercontent.com/78228205/198094147-4380e46c-71dc-445c-9fc1-be8c14ff7ef1.png)
* (some) Report Metrics from unbalanced dataset:<br>
![Screenshot from 2022-10-26 19-26-39](https://user-images.githubusercontent.com/78228205/198095057-c4ecdb57-83b0-4c4d-a10d-bd90997224e6.png)

<a name='3-balance'></a>
### 3. Balance the dataset by `product_category` and `sentiment`
1. Resampling: <br>
![Screenshot from 2022-10-26 19-43-34](https://user-images.githubusercontent.com/78228205/198098485-0f9d7097-2644-4ed8-82a5-cea3048270f7.png)
2. 'sentiment' by category: <br>
![Screenshot from 2022-10-26 19-48-37](https://user-images.githubusercontent.com/78228205/198099601-6d2d2d9f-aaa3-4bc6-b6c7-47b3651f87dd.png)

<a name='4-analyze'></a>
### 4. Analyze bias on balanced dataset with Amazon SageMaker Clarify
* Report category proportion from 'category_product' balanced dataset:
![Screenshot from 2022-10-26 19-52-56](https://user-images.githubusercontent.com/78228205/198100455-28d10193-150b-4532-80ec-4fcf1ea5abb4.png)

* (some) Report Metrics from balanced dataset:
![Screenshot from 2022-10-26 19-53-27](https://user-images.githubusercontent.com/78228205/198100466-dfd1d68c-cb26-4101-bbdc-c809197921a5.png)

          
