
metrics_list = [
    'precision',
    'recall',
    'accuracy',
    'f1' 
]

def print_cv_scores(model, X, y, cross_validator):
    from numpy import mean
    from numpy import std
    from sklearn.model_selection import cross_val_score
    for metric in metrics_list:
        score = cross_val_score(model, X, y, scoring=metric, cv=cross_validator)
        print(metric + ' %.3f (%.3f)' % (mean(score), std(score)))


def print_table_header():
    print ("Algorithm   \tPrecision\t\tRecall/TPR\t\tAccuracy\t\tF-Score")

def print_results_from_matrix(name, matrix):
    TP, FP = matrix[0]
    FN, TN = matrix[1]
    Precision = (TP * 1.0) / (TP + FP)
    Recall = (TP * 1.0) / (TP + FN)
    Accuracy = (TP + TN) * 1.0 /  (TP + TN + FP + FN)
    F_Score = 2*((Precision*Recall)/(Precision+Recall))
    print ("%s\t\t%.2f\t\t\t%.2f\t\t\t%.2f\t\t\t%.2f"%(name,Precision,Recall, Accuracy,F_Score))



