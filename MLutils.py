
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def cl(clf, x_train, x_test, y_train):
    ccc = clf
    ccc.fit(x_train, y_train)

    y_train_clf = ccc.predict(x_train)
    y_test_clf = ccc.predict(x_test)

    return y_train_clf, y_test_clf


def scores(clf, clfname, x_train, y_train, x_test, y_test, n_splits=3):

    # stability check using kCV, metric using accuracy
    kfold = KFold(n_splits)
    acc = cross_val_score(clf, x_train, y_train, cv=kfold, scoring='accuracy').mean()
    balacc = cross_val_score(clf, x_train, y_train, cv=kfold, scoring='balanced_accuracy').mean()
    print('Kfold Accuracy of {:s} classifier on training set: {:.2f}'
         .format(clfname, acc))
    print('Kfold Accuracy of {:s} classifier on test set: {:.2f}'
         .format(clfname, balacc))
    return acc, balacc


def confusion_scores(clf, X_test, y_test, clfname, outdir, tag='', saveFig=True):

    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, y_test_pred))

    confusion = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix: ", confusion)

    sns.heatmap(confusion, annot=True, cbar=False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if saveFig:
        tag = '_' + tag + '_'
        plt.savefig(outdir + clfname + tag + 'confusion_matrix.png')
    else:
        plt.show()


def ROC(clf, clfname, X_test, y_test, outdir, tag='', saveFig=True):
    """
     summarizes the model’s performance by evaluating the trade offs between
     true positive rate (sensitivity) and false positive rate(1- specificity)
    """

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

    plt.figure()
    plt.plot(fpr, tpr, label=clfname + ' (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if saveFig:
        tag = '_' + tag + '_'
        plt.savefig(outdir + clfname + tag + 'ROC.png')
    else:
        plt.show()



def GridSearch_logreg(X, y,
                      cv=3,
                      C=[1.0,1.5,2.0,2.5],
                      dual=[True, False],
                      # max_iter=100,
                      verbose=0, n_jobs=-1):


    param_grid = dict(dual=dual) # , max_iter=max_iter) #, C=C)
    from sklearn.linear_model import LogisticRegression

    from sklearn.model_selection import GridSearchCV
    gsc = GridSearchCV(
                    estimator=LogisticRegression(),
                    param_grid=param_grid,
                    cv=cv,
                    verbose=verbose,
                    n_jobs=n_jobs)

    grid_result = gsc.fit(X, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return gsc, grid_result


def GridSearch_SVMpoly(X, y,
                        nfolds=3,
                        Cs=[0.1, 1, 10],
                        degrees=[1, 2, 3],
                        gammas=[0.1, 1, 10],
                        verbose=0, n_jobs=-1):

    """

    - Kernel: The main function of the kernel is to transform the given dataset input data into the required form. There are various types of functions such as linear, polynomial, and radial basis function (RBF). Polynomial and RBF are useful for non-linear hyperplane. Polynomial and RBF kernels compute the separation line in the higher dimension. In some of the applications, it is suggested to use a more complex kernel to separate the classes that are curved or nonlinear. This transformation can lead to more accurate classifiers.

    - Regularization: Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. Here C is the penalty parameter, which represents misclassification or error term. The misclassification or error term tells the SVM optimization how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.

    - Gamma: (for non-linear). A lower value of Gamma will loosely fit the training dataset, whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting. In other words, you can say a low value of gamma considers only nearby points in calculating the separation line, while the a value of gamma considers all the data points in the calculation of the separation line.

    """

    param_grid = {'C': Cs, 'degree': degrees, 'gamma' : gammas}
    from sklearn.svm import SVC

    from sklearn.model_selection import GridSearchCV
    gsc = GridSearchCV(SVC(kernel='poly'),
                       param_grid,
                       cv=nfolds,
                       verbose=verbose,
                       n_jobs=n_jobs)

    grid_result = gsc.fit(X, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return gsc, grid_result


def GridSearch_RFC(X, y,
                    nfolds=3,
                    n_estimators=[5, 20, 100],
                    max_features=['auto', 'sqrt'],
                    max_depth=[10, 50, 100],
                    min_samples_split=[2, 5, 10],
                    min_samples_leaf=[1, 2, 4],
                    bootstrap=[True, False],
                    verbose=0, n_jobs=-1
                    ):
    """
        - n_estimators = number of trees in the foreset
        - max_features = max number of features considered for splitting a node
        - max_depth = max number of levels in each decision tree
        - min_samples_split = min number of data points placed in a node before the node is split
        - min_samples_leaf = min number of data points allowed in a leaf node
        - bootstrap = method for sampling data points (with or without replacement)
    """

    max_depth.append(None)

    param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    from sklearn.svm import SVC

    from sklearn.model_selection import GridSearchCV
    gsc = GridSearchCV(RandomForestClassifier(),
                       param_grid,
                       cv=nfolds,
                       verbose=verbose,
                       n_jobs=n_jobs)

    grid_result = gsc.fit(X, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return gsc, grid_result
