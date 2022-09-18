import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import model
import output

# inverse function to np.argmax
def convert_1d_to_2d(array):
    array_list = []
    cnt = 0
    for x in np.nditer(array):
        if x == 0:
            array_list.append(np.array([1,0])) 
        else:
            array_list.append(np.array([0,1]))
    cnt  = cnt + 1
    return array_list

# Converts Sparse Matrix to array
def sparse_matrix_to_array(matrix):
    X = matrix.toarray()
    return X


'''
Evasion attack based in projected gradient descent on logistic regression model
'''
def run_art_evasion_pgd(SKmodel, X_test, y_test, verbose):
    from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
    estimator = ScikitlearnLogisticRegression(model=SKmodel, clip_values=(0,1))
    y_test = y_test.to_numpy()

    # Benchmark

    # Evaluate the ART classifier on benign test examples
    y_bench = estimator.predict(X_test)
    # takes the higher value of each prediction array and makes 
    y_bench = np.argmax(y_bench, axis=1)
    # get confusion matrix and print results
    output.print_table_header()
    matrix = confusion_matrix(y_test, y_bench)
    output.print_results_from_matrix('ART_Benchmark_LGR', matrix)

    # Attack part

    input_variation = 0.1
    maximum_iterations = 1000
    b_size=100

    # Generate adversarial test examples
    from art.attacks.evasion import ProjectedGradientDescentNumpy
    attack = ProjectedGradientDescentNumpy(estimator=estimator, 
                                            eps=input_variation, 
                                            max_iter=maximum_iterations, 
                                            batch_size=b_size, 
                                            verbose=verbose)

    x_test_adv = attack.generate(sparse_matrix_to_array(X_test))

    # Evaluate the ART classifier on adversarial test examples
    y_pred = estimator.predict(x_test_adv)
    # takes the higher value of each prediction array and gives the index number (0 or 1)
    y_pred = np.argmax(y_pred, axis=1)

    matrix = confusion_matrix(y_test, y_pred)
    output.print_results_from_matrix('ART_Evasion_PGD', matrix)

'''
Decision Tree attack 
'''
def run_art_decision_tree_attack(SKmodel, X_train, y_train, X_test, y_test, verbose):
    from art.attacks.evasion import DecisionTreeAttack
    from art.estimators.classification import SklearnClassifier
    classifier_art = SklearnClassifier(SKmodel)
    y_test = y_test.to_numpy()

    '''
        Benchmark 
    '''
    # Evaluate the ART classifier on benign test examples
    y_bench = classifier_art.predict(X_test)
    # takes the higher value of each prediction array and makes 
    y_bench = np.argmax(y_bench, axis=1)
    # get confusion matrix and print results
    output.print_table_header()
    matrix = confusion_matrix(y_test, y_bench)
    output.print_results_from_matrix('ART_Benchmark_DTC', matrix)

    '''
        Attack part
    '''

    # Initialize attack
    attack = DecisionTreeAttack(classifier=classifier_art, offset=0.0001, verbose=verbose)
    # Generate adversarial samples
    print('DTC-Attack: Generating adversarial samples')
    x_adv = attack.generate(x=sparse_matrix_to_array(X_train))

    y_pred = classifier_art.predict(x_adv)
    # takes the higher value of each prediction array and gives the index number (0 or 1)
    y_pred = np.argmax(y_pred, axis=1)

    matrix = confusion_matrix(y_train, y_pred)
    output.print_results_from_matrix('ART_DecisionTree-Attack', matrix)

'''
SVM Poisoning attack
'''
def run_art_poisoning_svm(SKmodel, model_name, X_train, X_test, y_train, y_test, verbose):
    from art.estimators.classification.scikitlearn import ScikitlearnLinearSVC
    classifier = ScikitlearnLinearSVC(model=SKmodel, clip_values=(0.,1.0))

    # Evaluate the ART classifier on benign test examples
    y_pred = classifier.predict(X_test)
    # takes the higher value of each prediction array and gives the index number (0 or 1)
    y_pred = np.argmax(y_pred, axis=1)

    # get confusion matrix and print results
    matrix = confusion_matrix(y_test, y_pred)
    output.print_results_from_matrix('ART_Benchmark_SVC', matrix)

    # Attack part
    y_train = convert_1d_to_2d(y_train)
    y_test_svm = convert_1d_to_2d(y_test)

    y_train = np.array(y_train)
    y_test_svm = np.array(y_test_svm)
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    input_variation = 0.001
    maximum_iterations = 100
    s_step = 1.0

    # Generate adversarial test examples
    from art.attacks.poisoning import PoisoningAttackSVM
    try:
        attack = PoisoningAttackSVM(classifier=classifier, 
            eps=input_variation, x_train=X_train, y_train=y_train,
            step=s_step, x_val=X_test, y_val=y_test_svm, 
            max_iter=maximum_iterations, verbose=verbose)
        print('Attack generated with eps=[', input_variation, ']')
    except:
        print('Poisoning attack could not be initialized')

    attack_index = 6 # datapoint from which the algorith starts, this is arbitrary

    init_attack = np.copy(X_train[attack_index])
    y_attack = np.array([1,1]) - np.copy(y_train[attack_index])

    print('Starting poisoning attack')
    # Generate attack point and poisoned label
    attack_point, poisoned_labels = attack.poison(x=np.array([init_attack]),y=np.array([y_attack]))
    
    y_train = y_train.argmax(axis=1)
    y_train_poisoned = poisoned_labels.argmax(axis=1)

    # Add poisoned data to the dataset
    X_train_poisoned = np.concatenate((X_train, attack_point))
    y_train_poisoned = np.concatenate((y_train, y_train_poisoned))

    p_model = model.get_trained_model(model_option=model_name, X_train=X_train_poisoned, y_train=y_train_poisoned)

    # Evaluate the ART classifier on adversarial test examples
    y_pred_poisoned = p_model.predict(X_test)
    # Convert to array
    y_pred_poisoned = np.array(y_pred_poisoned)

    matrix = confusion_matrix(y_test, y_pred_poisoned)
    output.print_results_from_matrix('ART_Poisoning_SVM', matrix)

