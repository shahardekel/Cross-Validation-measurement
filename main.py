import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from warnings import filterwarnings


#A
def cross_validation_error(X,y,model,folds):
    average_train_error, average_val_error=0,0
    k_fold=KFold(n_splits=folds,shuffle=False)
    for train,test in k_fold.split(X):
        X_train=X[train]
        y_train=y[train]
        X_test=X[test]
        y_test=y[test]

        fit_model=model.fit(X_train,y_train)
        X_train_pred=fit_model.predict(X_train)
        X_test_pred = fit_model.predict(X_test)

        average_train_error+=1-accuracy_score(X_train_pred,y_train)
        average_val_error+=1-accuracy_score(X_test_pred,y_test)
    return [average_train_error/folds,average_val_error/folds]

#B
def Logistic_Regression_results(X_train, y_train, X_test, y_test):
    res_dict={}
    folds=5
    lambda_=[10**(-4),10**(-2),1,np.power(10,2),np.power(10,4)]

    for lam in lambda_:
        logreg = LogisticRegression(C=1/lam)
        cve_res=cross_validation_error(X_train,y_train,logreg,folds)
        fit_model = logreg.fit(X_train, y_train)
        test_pred = fit_model.predict(X_test)
        res_dict['LogReg_lam_' + str(lam)] = [cve_res[0], cve_res[1], 1 - accuracy_score(test_pred, y_test)]
    return res_dict

#C
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test,y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=7)
filterwarnings('ignore')

#D
Logistic_Regression_results_dict=Logistic_Regression_results(X_train,y_train,X_test,y_test)
#print(Logistic_Regression_results_dict)
average_train_error=[]
average_validation_error=[]
test_error=[]
for i in Logistic_Regression_results_dict:
    average_train_error.append(Logistic_Regression_results_dict[i][0])
    average_validation_error.append(Logistic_Regression_results_dict[i][1])
    test_error.append(Logistic_Regression_results_dict[i][2])


x = np.arange(len(average_train_error))
width=0.25
plot=plt.subplot()
plot1=plot.bar(x-width,average_train_error,width, label='Train Error')
plot2=plot.bar(x,average_validation_error, width,label='Validation Error')
pot3=plot.bar(x+width, test_error,width, label='Test Error')

for i in range(len(average_train_error)):
    plt.text(x =x[i]-0.38, y =average_train_error[i]+0.01, s = np.around(average_train_error[i],decimals=3), size = 6)
    plt.text(x=x[i]-0.1 , y=average_validation_error[i] + 0.01, s=np.around(average_validation_error[i], decimals=3), size=6)
    plt.text(x=x[i]+0.15, y=test_error[i] + 0.01, s=np.around(test_error[i], decimals=3), size=6)

plot.set_ylabel('Errors %')
plot.set_title('Errors By Different Lambdas On The Logistic Regression Model')
plot.set_xticks(x)
plot.set_xticklabels(tuple(Logistic_Regression_results_dict.keys()), fontsize=7)
plt.legend()
plt.show()


