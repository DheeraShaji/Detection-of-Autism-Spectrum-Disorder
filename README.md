
# Detection of Autism Spectrum Disorder using Machine Learning Techniques





## Abstract
Autism spectrum disorder (ASD) is a group of developmental disorder that has variant types and wide range of symptoms. ASD affected individuals face challenges in speech, interaction, verbal and non-verbal communication and behaviours. Till date, no clinical tests have been introduced that detects ASD. So, diagnosing ASD at its early stage is crucial to provide appropriate treatment to the patients. Thus, there is an urgent need for the development of easy and time- efficient detection of ASD. To analyze traits and improve the classification, three publicly available datasets in UCI repository is used, which is related to autism screening of children, adolescents and adults. Similar dataset of toddlers was taken from Kaggle. The datasets record behavioral and characteristic features of individuals. Previous studies on these data used various techniques to detect the presence of ASD, but the main down-side being its high latency which is a major drawback when it comes to actuality. The motto behind the project is to lower the latency making it usable in real-world scenarios. Machine learning supervised algorithms such as Support Vector Machines (SVM), Logistic Regression, k- Nearest Neighbours (kNN), Naïve Bayes and Random Forest are used. Chi- square feature selection technique is used to reduce the high dimensionality problem, that also reduces the time complexity. SVM gives the highest accuracy of 97.44 percentage with lowest execution time of 0.068 seconds.  

## Introduction
Autism Spectrum Disorder is a category of disease that affects how a person behaves, interact with others, how he/she feels and thinks. As the name suggests, it is a developmental disorder and can be diagnosed at any age. Usually, ASD symptoms are first identified in children of age one or two. There are wide range of symptoms and can be mild or severe and can vary from person to person.
Some of the common symptoms are
1. Language impairment
2. Attention deficit disorder
3. Repetitive behaviours
4. Being hyper active
5. Difficulty in social interaction
6. Mood disorders
7. Anxiety, etc.
Also, ASD affected individuals shows difficulty in controlling their emotions. It is difficult for them to pay attention and respond to situations. Some of them repeat their behaviours and some of them have difficulty in bringing a change to their daily life activities. As there is no cure for ASD, signs and symptoms can last throughout a patient’s life. Due to the time involved in diagnosing a patient and the increased number of affected patients, the healthcare domain has their own difficulties and economic impact. Thus, early detection is important as it can help to improve the quality of a patient’s lifestyle as well as it can help the healthcare professionals. With an early detection, doctors can prescribe adequate treatment, medication and therapy to the patients with a lot less time. Patients, their families and healthcare sector across the world is in need for a usable ASD detection technique that predicts if a patient is having ASD or not.
However, due to the regulations involved, collecting Autism data is hard. The data currently available is very less thus making it difficult to perform analysis on ASD.  

## Motivation
ASD, as a group of complex disorder has a prominent economic influence in the healthcare sector. This is because time involved in diagnosing the condition and the day-by-day increase in the number of ASD affected patients.
Diagnosing ASD can be difficult since there is no medical test, like blood test to diagnose the disorder. Early detection of ASD is very important as it allows the treatment or medication to start much earlier. Also, early detection can help doctors and health care workers to analyse, understand and let people know if they need to opt for medical analysis or not. Thus, there is an immediate requirement for time saving, easy and approachable Autism detection model globally.  

## Methodology
Methodology selected for this research is Cross industry process for data mining (CRISP- DM). It gives a structural and stable method for designing and implementing a machine learning project. Understanding the scenario:
The first step in creating any data mining project is to have a clear understanding on the ground knowledge in the area. In the field of ASD detection, only a few researches were carried out with the main motive being high accuracy. After studying the problem in the healthcare sector dealing with ASD, it was found that there is no proper ASD screening models that could be used effectively with real world cases. For scenarios in real-world, the main motive of should be to create a model with low latency and high accuracy.  
### 4.1	Data understanding:
In data mining projects, understanding the scenario is one of the most important aspect. At first glance,  
Toddler’s data consist of 1054 instances and 19 fields.  
Children’s data consist of 292 instances and 22 fields.  
Adolescents’ data consist of 104 instances and 22 fields.  
Adult’s data consist of 704 instances and 22 fields.  
Using Pearson correlation, statistical analysis was performed to better understand the data. Count of each feature and number of missing data was seen to better understand the data.
### 4.2	Data preparation:
As to produce an ML model that detects ASD for all age groups, the datasets of toddler’s, children, adolescents and adults were combined together before preprocessing. While performing statistical tests, it was noted that the independent features are not correlated to each other. Thus, it does not cause noise to the model.
 
Also, it was found that there is no class imbalance and biasness in the data as there are 1121 individuals diagnosed with ASD and 1033 individuals not diagnosed with ASD out of the total 2154 instances.
### 4.3	Modelling:
After performing categorical encoding, it was noticed that the problem of high dimensionality was present. To control this issue of high dimensionality, feature selection techniques were used. Chi-square feature selection technique is used in this project.
Various ML algorithms were implemented after the use of feature selection technique. Machine learning techniques such as Support Vector Machines (SVM), Logistic Regression, k-Nearest Neighbours (kNN), Naïve Bayes, Random Forest and Linear Discriminant Analysis are used in this research. In the end, to fine tune the machine learning model, hyperparameter tuning was conducted. Support Vector Machine, K-Nearest Neighbor Algorithm and Random Forest had hyperparameters which were tuned using Random Search whereas Nave Bayes algorithms had no parameters to optimize.
### 4.4	Evaluation:
Several evaluation metrices were used for the purpose of evaluation. Cross validation method was in use to examine the variance of the models. Accuracy, precision, recall, f1-score, and AUC are the evaluation metrices used to analyze the model performance.

## Design Specification
Free python Jupyter environment from Google known as Colab is used for the implementation of this project. For implementing machine learning projects, Colab is one of the well known cloud environment in research community. For ML projects, it supports many packages and one doesn’t need to install different packages. Sessions in Colab is equipped with either a GPU, CPU or TPU processor and a 13 GB of ram being run by a virtual machine. For Colab, the hardware specification is different for different projects. This project is executed on a ram of 13GB with 120Gb drive-space along with the aid of GPU processor.  

## Implementation
### 6.1	Data selection:
The dataset used in this project is publicly available and is downloaded from UPI repository.  
https://www.kaggle.com/fabdelja/autism-screening-for-toddlers/version/1  
https://archive.ics.uci.edu/ml/machine-learningdatabases/00419/ 
https://archive.ics.uci.edu/ml/machine-learningdatabases/00420/  
https://archive.ics.uci.edu/ml/machine-learningdatabases/00426/  

Four different datasets are used in this project, which is combined together before applying classification techniques. Table given below shows the detailed summary of the dataset.  
![table1](https://user-images.githubusercontent.com/78141360/142722598-78406cb8-0a09-417c-bd51-966b69c9cf19.JPG)
  Nineteen attributes are common for all the datasets but the attribute name is different in toddlers’ dataset. Before combining the datasets, the attributes names are made same in all the datasets. In toddlers’ dataset, the age is given in months whereas it is in years for all the other datasets. So, age in years is converted to age in months. The list of attributes in the combined dataset is shown in the table below.  
![table2](https://user-images.githubusercontent.com/78141360/142722595-219576b8-0be1-4ff9-abe6-327eef913a55.JPG)
As private data is not involved in the datasets used, there is no ethical issue with these data.
### 6.2	Exploratory data analysis:

The method of doing a primary investigation on the data is known as exploratory data analysis. This is done to find the pattern underlying in the data, to test the given hypothesis, to check the assumptions made and to spot anomalies. Before applying the machine learning models, the distribution of the data is checked so that their assumptions are satisfied.
### 6.3	Data sampling:

Imbalance problem occur where one of the two classes having more sample than other classes. Mainly in classification problems, class imbalance is considered to be a massive problem. It can affect the models and can result in biased output. Fig. 1 shows the bar plot that shows the count of both the classes in each dataset.  
Fig. 1: Count of classes in individual datasets.
![fig1](https://user-images.githubusercontent.com/78141360/142722592-366f0305-dfd1-4112-abe4-0876846f6a57.JPG)
To handle the problem of class imbalance, upsampling and downsampling are the two methods used. In simple words downsampling is reducing the majority data as to the level of minority class and upsampling is making new data for the marginalized class. The sampling technique applied in this project is Random downsampling. Here, the majority class is equalized to the size of minority class and thus the class imbalance problem is resolved. No additional libraries or pre-defined functions were used to perform random dowmsampling.
An analysis was also done combining all the four datasets. After combining, the problem of class imbalance was solved and there wasn’t any need to reduce the instances. Fig. 2 shows the count of both classes after combining into a single dataset.  
Fig. 2: Count of classes in combined dataset
![fig2](https://user-images.githubusercontent.com/78141360/142722591-a092702d-59a3-48bf-97cc-b29211175002.JPG)
### 6.4	Data pre-processing:

In machine learning, pre-processing is a major step involved. Absence of pre- processing can build on noise and can inturn influence the model being built. Almost all realistic data consist of unnecessary features, missing value fields, uncertain data, etc. Following are the preprocessing steps used in this project.
#### 6.4.1	Removing unwanted columns:

Unwanted features can add noise to the data and it is best advised to remove such columns. It was found that age_desc variable has the same value in all the fields. This means that it doesn’t contribute a lot to the target feature and using such features can create un-wanted noise.

#### 6.4.2	Imputing missing values:

As datasets with missing values could be biased, it is always advised to handle missing values in datasets. Also, data with missing values are not supported by most of the machine learning algorithms. Categorical columns with missing values were filled with most frequently occurring value whereas the missing values of numerical columns were filled with their mean.

### 6.5	Feature scaling:
 
Scaling techniques are applied to independent features to help normalize the data within a particular range of values. At times, scaling features also helps in speeding the calculations in ML algorithms. In this project, minmax scaler is used which estimates scales and transforms each feature to the new range.

### 6.6	Encoding categorical variables:

Only numerical values can be understood by machine learning models. So, encoding categorical features is an unavoidable step so that any ML models can be applied on to them. There are many ways of encoding and the ones used in this research project are One Hot encoding and label encoding.

#### 6.6.1	Label encoding:

If there are n categories, a value between 0 to n-1 is labelled by the label encoder. The same value gets assigned as before if a class repeats. The dependent variable is encoded in this project.

#### 6.6.2	One Hot encoder:

In One Hot encoding, we convert each categorical value into a new categorical column and assign 0 or 1 as values to those columns. A binary vector represents each integer value.

### 6.7	Feature selection:

Feature selection techniques are used to reduce the problem of high dimensionality. In this process, only important features are kept and the less contributing features gets removed. Numerous feature selection techniques exist these days. Some of the issues associated with high dimensionality are overfitting, unnecessary noise, high time complexity etc. Chi-square feature selection is used in this project.

#### 6.7.1	Chi-square feature selection:

The independence of events is checked in Chi-square feature selection. The two main events are presence of feature and that of class. Simply, it checks the independence of a particular feature with a class. If the events are not independent, we use that feature and the other way round.

### 6.8	Using Machine Learning Classification Algorithm:

#### 6.8.1	Random Forest:

Random forest consists of large number of decision tree that forms an ensemble. As all the tree-based models works well with categorical variables, random forest also works well with categorical features. As a classifier, it is well known for its accuracy [11], [12]. Random forest works well with an accuracy of 99 percent. But it is not considered the best model as it has high latency.
#### 6.8.2	Naïve Bayes:
 
In Naïve Bayes, Bayes theorem is being used. To work on categorical variables, Naïve Bayes is one of the best algorithms that can be used. It works wells in terms of both accuracy and low space complexity because it just calculates and stores probabilistic values.
Gaussian Naïve Bayes: It is a Naïve Bayes variant which uses Gaussian distribution. It worked with an accuracy of 57 percent which is the lowest of accuracy among all models.
Multinomial Naive Bayes: It is a variant of Naive Bayes that usese multinomial distribution. Multinomial Naive Bayes has better accuracy than that of Gaussian Naïve Bayes. It had an accuracy of 87 percent.

#### 6.8.3	K-Nearest Neighbours (kNN):

This algorithm works on the idea that points that are close to each other are similar. It makes no assumptions on the data provided and hence is known as non-parametric algorithm. In the training phase, the algorithm doesn’t learn anything, instead it stores the data and performs the required action during the classification process. Thus, it is known as lazy algorithm. kNN worked well with an accuracy of 95 percent.

#### 6.8.4	Linear discriminant analysis (LDA):

LDA is a classifier that uses Baye theorem and is made by fitting the class conditional density to the data. It is a classifier with linear decision boundary which is generated by fitting. Assuming that all classes share the same covariance matrix, the classifier model fits a Gaussian density to each existing class. Input dimensionality can also be reduced using this model by projecting it to the most discriminant directions. It shows an accuracy of 94.7 percent.

#### 6.8.5	Support vector machines (SVM):

A supervised ML algorithm which is established from the objective of locating the hyperplane that divides the dataset to two different class in the best possible way. This algorithm is named after support vectors that alter the location of the separating hyperplane. It shows an accuracy of 97 percent and has relatively less latency.

### 6.9	Hyper-parameter tuning:

For any machine learning model, optimizing the hyperparameter is known as hyperparameter tuning. Simply, it is an optimizing loop on models which are not tuned in the process of learning.

#### 6.9.1	Random search:

In this research, the hyperparameter tuning used is random search. In this tuning technique, random mix of parameters are used to get best set of parameters required for model building. Though it is like grid-search, in computing, random search performs better than it.
 
In our project, Random Forest, kNN and SVM are tuned using this technique. As Naïve Bayes algorithms doesn’t have any hyper-parameters, they were not tuned. kNN doesn’t comes up with much difference. SVM accuracy was boosted and showed an accuracy of 0.974 after training the model.

#### 6.10	Feature importance:

While working with supervised learning models, an important task is to determine the features that shows the best predicting powers. It is always useful to study the relationship between target feature and the most contributing features such that it can help us understand the underlying phenomenon much better. In this project, to predict whether an individual has ASD, we identified a small number of independent features that helps in strong prediction. It is expected that at the cost of performance, training time and prediction time will be lowered with only small number of contributing features to train with. Table. 3 shows the important feature from each dataset and that from the combined dataset.
Table. 3: Feature importance.  
![table3](https://user-images.githubusercontent.com/78141360/142722590-4c3d3e1c-85c0-4196-866b-59a2f8707416.JPG)
‘result’, ‘A9_Score’, ‘A5_Score’ and ‘A5_Score’ are the important features. Also, we see that the top most important feature ‘result’ (in term of their weightage factor) contribute heavily compared to the other features.

### 6.11	K-Fold cross validation:

K-fold cross validation divides the data into small subsets using statistical techniques. The basic approach used is to train the data on the subset created previously and checks its performance. This process is done for a pre-defined k number of times. This validation reduces variance as the model trains and tests on various data and hence reduce the variance. After cross validation, the mean of each iteration performance is considered as the result.

### 6.12	Evaluation:

A major part in any machine learning project is evaluation which show how models can perform according to various evaluation metrices. The metrices used in this project are:

#### 6.12.1	Accuracy:

A frequently used evaluation metric which simply is the correct prediction percentage for the test data. It is calculated using the following formula.

Accuracy = 𝑇𝑃+𝑇𝑁/(𝑇𝑃+𝐹𝑁+𝐹𝑃+𝑇𝑁)
 
#### 6.12.2	F1-Score

The harmonic mean of sensitivity and precision which tells us how many cases it has classified rightly. F1-score ranges from 0 to 1. The formula for F1 score is:
F1Score = 2*(Recall*Precision)/(Recall+Precision)

#### 6.12.3	Precision

Precision is the percentage of positive occurrence through total actual positive instances. Simply, when the model says it is right, precision tells how good the model is. The following formula calculates the same.
Precision = 𝑇𝑃/(𝑇𝑃+𝐹𝑃)

#### 6.12.4	Recall

Recall is defined as the percentage of positive occurrence through actual positive occurrences. Simply, it tells the true positive rate. The following formula calculates the same.

Recall = 𝑇𝑃/(𝑇𝑃+𝐹𝑁)

#### 6.12.5	AUC

Area under curve (AUC) provides an aggregate estimate of any model performance across all classification threshold. It finds which model predict in a better way which is often used in classification problem. As AUC does not bias on the size of evaluation data, it is considered better than accuracy as a classifier performance measure. It ranges from 0 to 1. It is calculated using specificity and sensitivity.

#### 6.12.6	Cohen’s Kappa:

Inter-rater reliability is measured using Cohen’s Kappa coefficient statistic. It tells if the model is randomly calculating the output. It ranges from -1 to 1 were
-1 denotes complete disagreement and 1 denotes complete agreement.  


## Result

Table. 4: Assessment of toddler’s data.  
![table4](https://user-images.githubusercontent.com/78141360/142722589-cbc11118-75d9-45db-bf1f-dc244ce34ad6.JPG)
Table. 4 shows the assessment details of toddler’s data. Random forest with and without hyperparameter tuning gave the highest cross validation of 100 percent. But the time elapsed while running this model is 1.211seconds, which is comparatively high. Whereas Support Vector Classifier with hyperparameter tuning has accuracy of 99.64 percent and time elapsed is 0.01 seconds.

Table. 5: Assessment of children’s data.  
![table5](https://user-images.githubusercontent.com/78141360/142722588-15e8fa1f-3a15-49a2-9f67-e34768c32f7e.JPG)
Table. 5 shows the assessment details of children’s data. Random forest with and without hyperparameter tuning gave the highest cross validation of 100 percent. But the time elapsed while running this model is 1.157 seconds, which is comparatively high. Whereas Support Vector Classifier with hyperparameter tuning has accuracy of 97.029 percent and time elapsed is 0.008 seconds.

Table. 6: Assessment of adolescents’ data.  
![table6](https://user-images.githubusercontent.com/78141360/142722587-fa1942e4-db4b-47a8-9156-ade2c123f5b1.JPG)
Table. 6 shows the assessment details of adolescents’ data. Random forest with and without hyperparameter tuning gave the highest cross validation of 100 percent. But the time elapsed while running this model is 1.059 seconds. Whereas Support Vector Classifier with hyperparameter tuning has accuracy of 95.23 percent and time elapsed is 0.003 seconds.

Table. 7: Assessment of adults’ data.  
![table7](https://user-images.githubusercontent.com/78141360/142722585-7497a464-a591-4bd0-8e09-2295c226f5ea.JPG)
Table. 7 shows the assessment details of adults’ data. Random forest with and without hyperparameter tuning gave the highest cross validation of 100 percent. But the time elapsed while running this model is 1.162 seconds, which is comparatively high. Whereas Support Vector Classifier with hyperparameter tuning has accuracy of 98.355 percent and time elapsed is 0.006 seconds.

Table. 8: Assessment of combined data.  
![table8](https://user-images.githubusercontent.com/78141360/142722583-36449cb1-70a9-49fa-90c8-4d922b907c55.JPG)
Table. 8 shows the assessment details of combined data. Random forest with hyperparameter tuning gave the highest cross validation score of 99.88 percent and F1- Score of 1 compared to other models. But the time elapsed while running this model is 1.711 seconds, which is comparatively high. Whereas Support Vector Classifier has an accuracy of 97.447 percent and the time elapsed is 0.06 8 seconds.
Considering the models built on individual datasets and the one build on combined dataset, SVM is considered the best model in this project.

## 8.	Discussion:

Challenges faced during this research project are as follows:
It is found that we need to have large datasets to build a low latency and accurate model. There is a need to understand and learn all the different parameters and their range before using a classifier. As most of the features are categorical and not numerical, we cannot claim that the model is optimum. With these datasets nothing can be improved as the model built is already at its best. Also, it is a tough task to collect ASD data with lot more information due to the existing rules and regulations. Thus, not much work has been done in this area. With this limitation under consideration, this research project resulted in building a detection model that can accurately detect the disorder within a feasible time frame.

## 9.	Conclusion:

The main aim of this project was to build an ASD detection model that can be used in real time without compromising on its accuracy. Several steps were done to attain these goals. At first, irrelevant features were removed to reduce the model complexity. Then, to reduce high dimensionality, feature selection was done, which also decreased complexity of the model and helped in increasing 85accuracy by dealing the noisy data. Finally, machine learning algorithms were used with hyper parameter tuning to achieve the best possible result.
To summarize, we applied various supervised machine learning algorithms to predict whether a given patient has ASD or not. We found that Multinomial Nave Bayes had the least execution time, but it was found to have an accuracy of 87 percent which is comparatively low. Support vector machines outperformed all the models and performs best in both the aspects of accuracy and that of low latency. This prediction surely can be of great help for the healthcare sector in screening ASD.
The test hypothesis is found to be true after building the model and the outputs shown in the results section shows the same. It was found that most of all the models have good accuracy but took lot of time in obtaining the output.
