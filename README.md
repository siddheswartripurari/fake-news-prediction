# fake-news-prediction

1.1	INTRODUCTION

This Machine Learning project is used to forecast the information whether it is fake or not based on the input given by the user. This Machine Learning project uses five different types of ML classifiers to forecast the data. The input given by the user must go to every classifier and after that, it will be forecast based on performance metrics. 

1.2 PROBLEM STATEMENT

The term “Fake News” was much less unknown and uncommon a few decades ago, but in the digital age of social media, it has grown into a massive monster. Fake news, filter bubbles, information manipulation and lack of trust in the media are growing problems within our society. However, addressing this issue requires a thorough understanding of fake news and its origins in the fields of Machine Learning (ML), Natural Language Processing (NLP), and Artificial Intelligence (AI) that could help us address this situation.

1.3 OBJECTIVE

The main objective of this project is to develop an expert system by using ML to detect fake news. Developing this model could help us to separate between which information is “Real” and which information is “Fake”.






1.4 ORGANIZATION OF REPORT

The remaining chapters of the project report are described as follows:

-	Chapter 2 gives us about literature survey, proposed system, and requirements.
-	Chapter 3 discusses the TF-IDF vectorizer and its advantages.
-	Chapter 4 gives the introduction of classifiers.
-	Chapter 5 tells us about performance metrics.
-	Chapter 6 shows the project framework and ER diagram.
-	Chapter 7 shows the pictures of project implementation.
-	Chapter 8 discusses the conclusion, future work, and references.



Chapter 2


2.1 LITERATURE SURVEY

There are several algorithms to detect fake news. For this, we analyze through different classifiers in different research papers. The classifiers are Random Forest, Convolutional Neural Network (CNN), Support Vector Machine (SVM), K-Nearest Neighbor (KNN), Logistic Regression, Naive Bayes, Long Short-Term Memory (LSTM), and SGD. The accuracy achieved using Random Forest is 85 %, the accuracy achieved by using CNN is 92%, the accuracy achieved by using SVM is 96%, the accuracy achieved by using KNN is 77%, the accuracy achieved by using Logistic Regression is 95%, the accuracy achieved by using Naive Bayes is 90%, the accuracy achieved by using Long Short-term memory is 97%, the accuracy achieved by using the combination of SVM &NB is 78% and the accuracy achieved by using SGD is 77.2%. Compared with all CNN, LR and LSTM obtain high accuracy.

2.2 PROPOSED SYSTEM

This project's main purpose is to develop an expert system to predict the Fakes by using different types of classifiers (Linear Regression, Decision Tree, Passive Aggressive, Gradient Boosting, Random Forest), and using TF-IDF vectorization algorithm (we can see about this in Chapter 3) and lastly manual testing.

2.3 METHODOLOGY

First, we need to train our model. For that, I took two data sets from the Kaggle website. One data set named “fake.csv” contains fake data text and another data set named “true.csv” contains true data text. Before we train the model, we need to clean both datasets first because our dataset may contain incomplete data. Data cleaning is the most important part. If we didn’t cleanse the data, it may affect the accuracy result, and the model we are going to design shows inaccurate results. After cleansing the data apply the TF-IDF vectorizer. This algorithm transforms text into a meaningful representation of numbers which is used to fit the machine algorithm for prediction. After transforming apply classifiers for prediction. For manual testing, the data is taken through the read method, or we can select the data from our files also. 

2.4 HARDWARE DETAILS

To build and run a model, Jupyter Notebook or Google Colaboratory is required and GPU is necessary too because we are dealing with a huge amount of data, and it takes a lot of time to run if we utilize CPU. In Google Colaboratory we can change our runtime type to GPU. So, I will suggest Google Colaboratory.


2.5 SOFTWARE DETAILS

RAM – Min. 4GB or More 
DISK – Min. 2GB or More
Google Computing Engine (Backend)
These requirements are all automatically allocated in Google Colaboratory when you try to create a new project.




Chapter 3
TF-IDF VECTORIZER


3.1 TF-IDF VECTORIZER

TF (Term Frequency): The number of times a word appears in a document is its Term Frequency. A higher value means that a term appears more often than others, and so the document matches well when the term is part of the search terms.

IDF (Inverse Document Frequency): Words that appear more than once in a document, but also several times in many others may not be relevant. IDF is a measure of the meaning of a term in the entire corpus.

The Tfidf Vectorizer converts a collection of raw documents into a meaningful representation of numbers in a matrix of TF-IDF features.


3.2 ADVANTAGES OF TF-IDF VECTORIZER

Advantages of TF-IDF:
•	 TF-IDF is a simple and efficient algorithm for matching words in a query to documents that are relevant to the query. 
•	The research done by many of the scholars till now has proven that TF-IDF returns documents that are highly relevant to a particular query. 
•	Over the years, TF-IDF has formed the basis of all the research that has been carried out on the development of document query algorithms.





Chapter 4

ALGORITHMS


4.1 MACHINE LEARNING CLASSIFICATION

Machine learning (ML) is a class of algorithms that help software systems achieve more accurate results without having to reprogram them directly. Data scientists characterize the changes or characteristics that the model needs to analyze and utilize to develop predictions. When the training is complete, the algorithm breaks down the learned levels into new data. Five algorithms are adopted in this document to classify fake news.



4.2 CLASSIFICATION ALGORITHMS

4.2.1 LOGISTIC REGRESSION

It is a machine learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes and success) or 0 (no and failure).
Since we classify text based on a large feature set, with a binary output (true/false or true article / fake article), a logistic regression (LR) model is used, as it provides the intuitive equation to classify problems into binary or multiple classes.
 
Fig 4.1 – Logistic Regression for ML

4.2.2 DECISION TREE

A decision tree is an important tool that works in a structure like a flowchart, used primarily for classification issues. Each internal decision tree node specifies a condition or “test” on an attribute and the branching is made based on the test conditions and the result. Finally, the leaf node has a class label which is obtained after calculating all the attributes. The distance from the root to the leaf represents the classification rule. The amazing thing is that it can work with category and dependent variables. They are useful for identifying the most important variables and they also describe the relationship between the variables quite appropriately. They are important in creating new variables and features which is useful for data mining and predicting the target variable quite efficiently.

 
Fig 4.2 – Decision Tree Classifier Structure
4.2.3 RANDOM FOREST

Random Forest is built on the concept of building many decision tree algorithms, after which the decision trees get a separate result. The results, which are predicted from many decision trees, are taken from the random forest. To ensure variation in decision trees, Random Forest randomly selects a subcategory of properties from each group. The applicability of Random Forest is best when used on uncorrelated decision trees. When applied to similar trees, the overall result will be more or less similar to a single decision tree. Uncorrelated decision trees are often obtained by bootstrapping and have randomness.

 
Fig 4.3 – Random Forest Classifier structure


4.2.4 PASSIVE-AGGRESSIVE

Passive Aggressive algorithms are online learning algorithms. This algorithm remains passive for a correct classification result and becomes aggressive in the event of calculation, update, and adjustment errors. Unlike most other algorithms, it doesn’t converge. Its purpose is to make updates that correct the loss, causing a minimal change in the norm of the weight vector.



4.2.5 GRADIENT BOOSTING

Gradient Boosting is a machine learning technique for regression, classification, and other activities, which produces a prediction model in the form of a set of weak prediction models, usually decision trees. When a decision tree is a weak learner, the resulting algorithm is called gradient-boosted trees, which generally outperform the random forest. It builds the model gradually as do other improvement methods and generalizes them allowing the optimization of an arbitrary differentiable loss function.



