# *Don't get kicked*
The project is carried out as part of the Research Workshops subject, majoring in Engineering and Data Analysis at the Faculty of Mathematics and Information Sciences of the Warsaw University of Technology, 
under the substantive supervision of **mgr inż. Mateusz Chiliński**.

**Authors**: Maciej Borkowski, Mateusz Kubita, Tymoteusz Kwieciński

# Description

Our main goal was to achieve the best possible result in the kaggla competition, and thus to build a model predicting whether the purchase of a given car would be risky. 
In other words, our goal was to predict whether the buyer of the given car can lose a considerable amount of money on this purchase

We were given the dataset containing information about car purchased on auctions in USA during years 2007-2008. There are multiple features of the cars included in the dataset such as its age, condition, price on different auctions etc.

# Methods applied to solve the problem

We focused on building the optimal machine learning predicting the target feature - that is, answer the question whether the car is a bad buy. We used many popular classification models such as Random Forest, SVM, XGBoost, etc. 
Apart from model building, exploratory data analysis and preprocessing the given training dataset, we also tried to overcome some various problems related to the specificity of the problem. 
As we noticed, in the dataset there was strong imbalance between the predicted classes - the class of bad buys was severely underrepresented. What is more we decided that we should focus more on reducing the number of False Negative predictions - it is better not to buy a suspicious car than to buy it and lose a lot of money.
We also tried to make our models as explainable as possible.

# Used software

All ours solutions were written in Python. We extensively used the `scikit-learn` library - mostly to train models and score their performance. In order to properly handle tabular data we used libraries such as `pandas` and `numpy`.
An explanation and comparison of our models as well as a summary of our work required usage of plotting libraries - `matplotlib`, `seaborn` and `shap` were used. In addition, we used the library `pickle` to save the models.
What is more we used also few fundamental libraries such as `os`, `warnings` and `sys`. 

