######################################### Function to calculate confusion matrix score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def confusionMatrix_analysis(X, y, m):
    ypred = m.predict(X) # calculate ypred using model m
    print(confusion_matrix(y, ypred)) # show confusion table for both ytest with ypred
    """
    top_left (TN); bottom_right (TP); bottom_left (FN); top right (FP)
    TP : surviving passenger correctly predicted
    TN : drowned passenger correctly predicted
    FP : drowned passenger predicted as surviving
    FN : surviving passenger predicted as drowned
    """
    print("Precision score: " + str(precision_score(y, ypred))) #Note: ytest first, after the ypred
    print("Recall score: " + str(recall_score (y, ypred)))
    #ROC curve

######################################### Function to calculate bootstrapping
from sklearn.utils import resample
def bootstrapping_analysis(X, y, m):
    boots = []
    for i in range(1000):
        Xb, yb = resample(X, y)
        m.fit(Xb, yb)
        score = m.score(Xb, yb)
        boots.append(score)
        #print(i, score)

    # get percentiles for 90% confidence
    boots.sort()
    ci80 = boots[100:-100]
    print(f"80% confidence interval: {ci80[0]:5.2} -{ci80[-1]:5.2}")
    ci90 = boots[50:-50]
    print(f"90% confidence interval: {ci90[0]:5.2} -{ci90[-1]:5.2}")
    ci95 = boots[25:-25]
    print(f"95% confidence interval: {ci95[0]:5.2} -{ci95[-1]:5.2}")
    ci99 = boots[5:-5]
    print(f"99% confidence interval: {ci99[0]:5.2} -{ci99[-1]:5.2}")

######################################### Function to calculate cross validation
from sklearn.model_selection import cross_val_score
def bootCrossVal_analysis(Xtrain, Xtest, X, ytrain, ytest, y, m):
    # Bootstrap analysis:
    bootScoreTrain = bootstrapping_analysis(Xtrain, ytrain, m) # calculate bootstrapping score
    #bootScoreTest = bootstrapping_analysis(Xtest, ytest, m) # calculate bootstrapping score
    bootScoreAll = bootstrapping_analysis(X, y, m) # calculate bootstrapping score
    print("Bootstrap score (train data): " + str(bootScoreTrain))
    print("Bootstrap score (all data): " + str(bootScoreAll))

    # Cross-validation analysis:
    crossValScoreTrain = cross_val_score(X=Xtrain, y=ytrain, estimator=m, cv=5)
    #crossValScoreTest = cross_val_score(X=Xtest, y=ytest, estimator=m, cv=5)
    crossValScoreAll = cross_val_score(X=X, y=y, estimator=m, cv=5)
    print("Cross-validation score (train data): " + str(crossValScoreTrain))
    print("Cross-validation score (all data): " + str(crossValScoreAll))

################ TEST MULTIPLE HYPERPARAMETERS


#X, y = load_iris(return_X_y=True)
#depths = [2, 3, 4, 5]
#trees = [2, 20, 40, 80, 100]
####Use GridSearch to loop across multiple parametters ####
from sklearn.model_selection import GridSearchCV

def testMultipleHyperParameters(X, y, method, n_estimators, depths, scoring, n_jobs, cv):
    if method == 'RandomForest':
        print('\n##### RandomForest GridSearch #####')
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier()
        grid = GridSearchCV(m,
                            param_grid = {'n_estimators': n_estimators,
                            'max_depth': depths},
                            scoring= scoring,
                            n_jobs= n_jobs,
                            cv=cv
                            )
        grid.fit(X, y)
        grid.best_estimator_
        print('Best parameter (' + method + '): '+ str(grid.best_params_))
        print('Best score (' + method + '): ' + str(grid.best_score_) )

    elif method == 'LogisticRegression':
        print('\n##### LogisticRegression GridSearch #####')
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(solver='lbfgs', max_iter=1000)
        grid = GridSearchCV(m,
                            param_grid = {'C': n_estimators},
                            scoring= scoring,
                            n_jobs= n_jobs,
                            cv=cv
                            )
        grid.fit(X, y)
        grid.best_estimator_
        print('Best parameter (' + method + '): '+ str(grid.best_params_))
        print('Best score (' + method + '): ' + str(grid.best_score_))

    else:
        print('Atention: Select classification method!')

        return grid.best_estimator_, grid.best_params_, grid.best_score_
   



