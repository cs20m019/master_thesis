def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict_model(model, X_test): 
    prediction = model.predict(X_test)
    return prediction

'''
    When training an SVM with the Radial Basis Function (RBF) kernel, two parameters must be considered: C and gamma. 
    The parameter C, common to all SVM kernels, trades off misclassification of training examples against simplicity of the decision surface. 
    A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly. gamma defines how much influence a single training example has. 
    The larger gamma is, the closer other examples must be to be affected.
''' 

max_iterations = 5000

#
def get_model(model_option): 
    if model_option == 'LGR':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver='lbfgs', max_iter=max_iterations)
    elif model_option == 'linear_SVC':
        from sklearn.svm import LinearSVC
        model = LinearSVC(max_iter=(max_iterations*2))
    # SVM with Radial Basis Function (rbf), C=1.ß, gamme='scale' (uses 1 / (n_features * X.var()) as value of gamma) 
    elif model_option == 'rbf_SVC':
        from sklearn.svm import SVC
        model = SVC(C=1.0, gamma='scale', kernel='rbf')
    elif model_option == 'DTC':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    # Multi-layer Perceptron, 
    elif model_option == 'MLP':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(solver='lbfgs', max_iter=(max_iterations/5), warm_start=True)
    else:
        pass
    return model

# run model functions and prints the results
def get_trained_model(model_option, X_train, y_train): 
    model = get_model(model_option)
    model = train_model(model=model, X_train=X_train, y_train=y_train)
    return model


def get_confusion_matrix(model, X_testdata, y_testdata):
    from sklearn.metrics import confusion_matrix
    y_pred = predict_model(model=model, X_test=X_testdata)
    matrix = confusion_matrix(y_testdata, y_pred)
    return matrix

def get_prediction(model, X_testdata):
    y_pred = predict_model(model=model, X_test=X_testdata)
    return y_pred



