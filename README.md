# Risk Prediction for Home Equity Line of Credit(HELOC)

----
## Project Outline
We develop a predictive model and a decision support system(DSS) that evaluates the risk of Home Equity Line of Credit (HELOC) applications with Streamlit.

### Data
The dataset and additional information can be found on [here](https://community.fico.com/s/explainable-machinelearning-challenge). This also excel sheets
along with its decision rules

### Details
Using the dataset, we developed a predictive model to assess credit risk. We compared various models and different algorithms and shortlisted the most appropriate algorithm for prediction depending upon the accuracy. We have also created a prototype of an interactive interface that sales representatives in a bank/credit card company can use to decide on accepting or rejecting applications.

## Approach
### Data cleaning
Firstly, we see the percentage of pf -7, -8, -9 of the total rows. We drop the rows containing special values such as -9. As we have a significant number of values of -7 & -8, dropping those values would result in less number of datapoints which would result in in accurate evaluation of our models. Besides, if we take average of these special values, it would result in complete change of original meaning of these values. Hence, we could not move forward with same. Thus, post initial research we can say that the special values -7, -8 does have certain relevance in order to arrive at our results. Further, we separated Independent (X) variables from Independent Variables (Y). We considered all the variables except RiskPerformance as an
independent variables and Risk Performance was the predicted variable.

Post separation, we had reconstructed the variables Max Delqz Public’/ ‘Max Delq Ever’ using get_dummies and we were able to recreate Independent Variables. We do so because the category represents delinquency status and not all values are numeric. They are majorly categorical variables.

### Modeling and evaluation
We used various models to create prediction and followed by certain parameters such RSME and accuracy to evaluate these models. Some of the models included SVM, Logistic Regression KNN, Gaussian, Decision Tree classifier, Random Forest, Ada Boost classifier. We had evaluated these models on the lines of methods we learned during our classes. We had splitted
the data set into training and testing data on a loose ratio of 75:25 split.

Moving forward, we used Hyper parameter tuning using GridSearch CV in order to find the best model among all. We created data frame which containing each of the model. Post this, we calculated the values of Root Mean Square error, accuracy, recall, precision and CV score to choose the best model having best accuracy and least error.

Lastly, we summarized the whole models and checked the value of best model

As we see from below, the accuracy of **Ada Boost** is highest which is 0.7287 and that the RMSE is the the lowest which is 0.520855 among other models. We also took into consideration Precision and Recall for arriving at better results.

Model Comparison: ![model](/image/comparison.png)

## Interface
An interface was developed using in order to represent respective models. We did data cleaning using pipeline and stream lit package for the interface.
We can input the row of information in the dataset using the side bars or the user can select the one of the rows in the data frames to make the prediction.
![interface](/image/streamlit.png)

![interface](/image/streamlit2.png)
