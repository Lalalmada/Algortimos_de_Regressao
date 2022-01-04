
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score # Cross Validation Function.
from sklearn.model_selection import KFold # KFold Class.
from sklearn.linear_model import LinearRegression # Linear Regression class.

df = pd.read_csv("../datasets/Admission_Predict.csv")

df.drop('Serial No.', axis = 1, inplace = True)

x = df.drop('Chance of Admit ', axis = 1)
y = df['Chance of Admit ']

model  = LinearRegression()
kfold  = KFold(n_splits=5, shuffle=True) # shuffle=True, Shuffle (embaralhar) the data.
result = cross_val_score(model, x, y, cv = kfold)

print("K-Fold (R^2) Scores: {0}".format(result))
print("Mean R^2 for Cross-Validation K-Fold: {0}".format(result.mean()))


#### OUTPUT
##### K-Fold (R^2) Scores: [0.75872602 0.84426748 0.67785048 0.82651749 0.82568233]
#####Mean R^2 for Cross-Validation K-Fold: 0.7866087616458846

# #Ótimo, agora temos o nosso R² para K iterações com dados de treinos e teste aleatórios… Agora vem a pergunta: 
# Como eu posso criar uma função que veja a performance (R²) de vários modelos (Ex: Regressão) e escolha o melhor?

def ApplyesKFold(x_axis, y_axis):
  # Linear Models.
  from sklearn.linear_model import LinearRegression
  from sklearn.linear_model import ElasticNet
  from sklearn.linear_model import Ridge
  from sklearn.linear_model import Lasso

  # Cross-Validation models.
  from sklearn.model_selection import cross_val_score
  from sklearn.model_selection import KFold

  # KFold settings.
  kfold  = KFold(n_splits=10, shuffle=True) # shuffle=True, Shuffle (embaralhar) the data.

  # Axis
  x = x_axis
  y = y_axis

  # Models instances.
  linearRegression = LinearRegression()
  elasticNet       = ElasticNet()
  ridge            = Ridge()
  lasso            = Lasso()

  # Applyes KFold to models.
  linearRegression_result = cross_val_score(linearRegression, x, y, cv = kfold)
  elasticNet_result       = cross_val_score(elasticNet, x, y, cv = kfold)
  ridge_result            = cross_val_score(ridge, x, y, cv = kfold)
  lasso_result            = cross_val_score(lasso, x, y, cv = kfold)

  # Creates a dictionary to store Linear Models.
  dic_models = {
    "LinearRegression": linearRegression_result.mean(),
    "ElasticNet": elasticNet_result.mean(),
    "Ridge": ridge_result.mean(),
    "Lasso": lasso_result.mean()
  }
  # Select the best model.
  bestModel = max(dic_models, key=dic_models.get)

  print("Linear Regression Mean (R^2): {0}\nElastic Net Mean (R^2): {1}\nRidge Mean (R^2): {2}\nLasso Mean (R^2): {3}".format(linearRegression_result.mean(), elasticNet_result.mean(), ridge_result.mean(), lasso_result.mean()))
  print("The best model is: {0} with value: {1}".format(bestModel, dic_models[bestModel]))


if __name__ =='__main__':
  import pandas as pd

  df = pd.read_csv("../datasets/Admission_Predict.csv")
  df.drop('Serial No.', axis = 1, inplace = True)

  x = df.drop('Chance of Admit ', axis = 1)
  y = df['Chance of Admit ']

  ApplyesKFold(x, y)
  
  
  ## OUTPUT
  
 ## Linear Regression Mean (R^2): 0.7897084153867848
 ## Elastic Net Mean (R^2): 0.5203451094489794
 ## Ridge Mean (R^2): 0.7771667506690682
 ## Lasso Mean (R^2): 0.2569904339264844
 ## The best model is: LinearRegression with value: 0.7897084153867848
 ## Linear Regression Mean (R^2): 0.7866950600812153
 ## Elastic Net Mean (R^2): 0.532657928647275
 ## Ridge Mean (R^2): 0.7876746996784943
 ## Lasso Mean (R^2): 0.25087272330850907
 ## The best model is: Ridge with value: 0.7876746996784943
 ## Linear Regression Mean (R^2): 0.7881860034693391
 ## Elastic Net Mean (R^2): 0.5316122498998679
 ## Ridge Mean (R^2): 0.7855190176462508
 ## Lasso Mean (R^2): 0.2549741277785563
The best model is: LinearRegression with value: 0.788186003469339
