
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
    print("Confusion Matrix":, confusion)

    sns.heatmap(confusion, annot=True, cbar=False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if saveFig:
        tag = '_' + tag + '_'
        plt.savefig(outdir + clfname + tag + 'confusion_matrix.png')
    else:
        plt.show()


def ROC(clfname, X_test, y_test, outdir, tag='', saveFig=True):
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



def GridSearch_logreg(X, y, max_features=None,
                      n_estimators=[10, 20, 100, 1000],
                      cv=3,
                      C = [1.0,1.5,2.0,2.5],
                      dual = [True, False],
                      verbose=0, n_jobs=-1):


    param_grid = dict(dual=dual, max_iter=max_iter) #, C=C)

    from sklearn.model_selection import GridSearchCV
    gsc = GridSearchCV(
                    estimator=LogisticRegression(),
                    param_grid=param_grid,
                    cv=cv,
                    verbose=verbose,
                    n_jobs=n_jobs)

    grid_result = gsc.fit(X, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params))
    return gsc, grid_result