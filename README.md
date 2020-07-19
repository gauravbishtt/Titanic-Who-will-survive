# Titanic-Who-will-survive

ABSTRACT: The following project predicts the survival of passengers in the infamous ship titanic.

DATASET: The training-set has 891 examples and 11 features + the target variable (survived). 2 of the features are floats, 5 are integers and 5 are objects. Below I have listed the features with a short description: 
Survival: Survival 0 or 1 
PassengerId: Unique Id of a passenger. 
pclass:Ticket class
sex:Sex
Age:Age in years
sibsp:number of siblings or spouses aboard the Titanic
parch:number of parents or children aboard the Titanic
ticket:Ticket number
fare:Passenger fare
cabin:Cabin number
embarked:Port of Embarkation 

DATA ANALYSIS AND PREPROCESSING: Feature conversion of data types from float to int64 and spotting of features containing missing value and replacing them with values.Analysing on which features the survival is more and removing of unessecary features. ALGORITHMS: As it is classification problem we have split the dataset into training and testing set and trained it on various models and got the best accuracy in Random forest classifier which is= 96.3
