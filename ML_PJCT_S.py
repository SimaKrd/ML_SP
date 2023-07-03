import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Load data
data_Train = pd.read_csv("/content/Churn_Modelling.csv",sep=';')
data_Test = pd.read_csv("/content/Churn_Modelling_scoring.csv",sep=";")

# Check for missing values in the training data
print(data_Train.isna().sum())

# Prepare the training data
data = data_Train.iloc[:, 1:12]
Target = data_Train.iloc[:, 12]

# Extract categorical and numerical features
col_catg = [2,3]
col_numr = [0,1,4,5,6,7,8,9,10]

X_col_catf = data.iloc[:, col_catg].values
X_col_numr = data.iloc[:, col_numr].values

# One-hot encode categorical features
enc = OneHotEncoder()
X_Cat_encoded = enc.fit_transform(X_col_catf).toarray()

# Standardize numerical features
scaler = StandardScaler()
X_col_numr_scaled = scaler.fit_transform(X_col_numr)

# Concatenate features
X_train = np.concatenate((X_col_numr_scaled, X_Cat_encoded), axis = 1)

# Apply PCA
pca = PCA(n_components=2)
X_train_Pca = pca.fit_transform(X_train)
X_train_merged = np.concatenate((X_train_Pca, X_train), axis = 1)

# Prepare the testing data
X_col_catf_test = data_Test.iloc[:, col_catg].values
X_col_numr_test = data_Test.iloc[:, col_numr].values

X_Cat_encoded_test = enc.transform(X_col_catf_test).toarray()

X_col_numr_scaled_test = scaler.transform(X_col_numr_test)

X_test = np.concatenate((X_col_numr_scaled_test, X_Cat_encoded_test), axis = 1)
X_test_Pca = pca.transform(X_test)
X_test_merged = np.concatenate((X_test_Pca, X_test), axis = 1)

# Custom cost function
def cost_function(y_true, y_pred):
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    cost = 0.5 * recall_class_0 + 0.5 * recall_class_1
    return cost

# GridSearchCV on MLPClassifier
param_grid_mlp = {
    'max_iter' : [200,230,100,150],
    'hidden_layer_sizes' : [[40],[8,10],[14,7],[30,14]],
    'activation' : ['identity', 'tanh', 'relu'],
    'random_state' : [1]  
}

grid_mlp = GridSearchCV(MLPClassifier(), param_grid_mlp, cv=5, scoring = cost_function)
grid_mlp.fit(X_train_merged, Target)
print("Best MLP parameters: ", grid_mlp.best_params_)
print("Best MLP score: ", grid_mlp.best_score_)

# GridSearchCV on RandomForestClassifier
param_grid_rfc = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_rfc = GridSearchCV(RandomForestClassifier(random_state=1), param_grid_rfc, cv=5, scoring = cost_function)
grid_rfc.fit(X_train_merged, Target)
print("Best RFC parameters: ", grid_rfc.best_params_)
print("Best RFC score: ", grid_rfc.best_score_)

# GridSearchCV on KNeighborsClassifier
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring = cost_function)
grid_knn.fit(X_train_merged, Target)
print("Best KNN parameters: ", grid_knn.best_params_)
print("Best KNN score: ", grid_knn.best_score_)

# Make predictions with the best model and export to CSV
model = MLPClassifier(activation= 'tanh', hidden_layer_sizes = [18,10, 10], max_iter= 150, random_state = 1)
model.fit(X_train_merged, Target)
resulat = model.predict(X_test_merged)

Customer_Id = data_Test.iloc[:, 1]
df = pd.DataFrame({"Customer_Id": Customer_Id, "Result": resulat})
df.to_csv('scores.csv', index=False)







