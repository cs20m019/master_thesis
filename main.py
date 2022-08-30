import data_PrEP as DP
import model as MOD
import output as OUT
import usecase_art as UCART
import os
from time import process_time

# global variables
t_size = 0.2
r_state = 4

# Number of records that is randomly taken from the CSIC2010 dataset
dataframe_size = 1000

# Vectorize parameters
analyzer = 'char'
ngram_range = (3,8)

# Number of splits for kfold cross-validation
n_splits = 5

model_list = [
    'LGR', 
    'DTC',
    #'linear_SVC',
    'rbf_SVC',
    #'MLP'
]

def run_performance_evaluation(sklearn_vectorizer, X, y):
    # Get cross-validator to account for overfitting
    cv = DP.get_crossvalidator_kfold(split_number=n_splits)

    # Vectorize data
    X = DP.vectorize_document(sklearn_vectorizer, analyzer, ngram_range, X)
    X = DP.normalize_data(X)
    # Scale copy  with MaxAbs which is recommended for sparse-matrix
    X_scaled = DP.max_abs_scale_data(X)

    if sklearn_vectorizer == 'count':
        print('Running performance evaluation for CountVectorizer')
    elif sklearn_vectorizer == 'tfid':
        print('Running performance evaluation for tfid_Vectorizer')
    elif sklearn_vectorizer == 'hash':
        print('Running performance evaluation for HashingVectorizer')
    else: 
        print('Could not run performance evaluation' )
        print('[ '+ sklearn_vectorizer + ' ] is not a supported vectorizer')
        exit()

    # Performance metrics are calculated for both normal and scaled data with cross-validation
    for model_name in model_list:
        # get sklearn-model 
        model = MOD.get_model(model_option=model_name)
        print('Normal ', model_name)
        ts1_start = process_time()
        OUT.print_cv_scores(model=model, X=X, y=y, cross_validator=cv)
        ts1_stop = process_time()
        print('Time elapsed for normal ', model_name , ' : [ ' , (ts1_stop-ts1_start) ,' ] seconds\n')
        print('Scaled ', model_name)
        ts2_start = process_time()
        OUT.print_cv_scores(model=model, X=X_scaled, y=y, cross_validator=cv)    
        ts2_stop = process_time()
        print('Time elapsed for scaled ', model_name , ' : [ ' , (ts2_stop-ts2_start) ,' ] seconds\n')


# Performs predefined adversarial attacks on machine learning models and provides performance data pre-attack and post-attack
def run_adversarial_attack(attack_type, sklearn_vectorizer, X, y, verbose):
    # Vectorize data
    X = DP.vectorize_document(sklearn_vectorizer, analyzer, ngram_range, X)
    X = DP.normalize_data(X)
    # Scale copy  with MaxAbs which is recommended for sparse-matrix
    X = DP.max_abs_scale_data(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=r_state)

    if attack_type == 'PGD':
        print('Running PGD-Attack...')
        UCART.run_art_evasion_pgd(MOD.get_trained_model('LGR', X_train=X_train, y_train=y_train ), X_test, y_test, verbose)
    elif attack_type == 'DT-Attack':
        print('Running Decision-Tree Attack...')
        UCART.run_art_decision_tree_attack(MOD.get_trained_model('DTC', X_train=X_train, y_train=y_train),X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, verbose=verbose)
    elif attack_type == 'SVM_Poisoning':
        print('Running SVM Poisoning Attack...')
        UCART.run_art_poisoning_svm(MOD.get_trained_model('rbf_SVC', X_train=X_train, y_train=y_train), model_name='rbf_SVC', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, verbose=True)
    else:
        print('Could not run this attack' )
        print('[ '+ attack_type + ' ] is not a supported adversarial attack')
        exit()

# Run the full script
def run_full_thesis_script():
    print('Starting script...')

    from datetime import datetime
    # Write the dataframe to a file for documentation and reprocessing purposes
    now = datetime.now()
    str_dataframe_size = str(dataframe_size)
    # Create filepath YYYYMMDD_dataframesize.csv
    file_path = now.strftime('%Y%m%d') + '_' + str_dataframe_size +'.csv'

    # If a subset of size datafram_size was already created reopen it for traceability purposes
    if not os.path.exists(file_path):
        # Get CSIC2010 dataframe
        df = DP.get_csic2010_dataframe(dataframe_size)
        df.to_csv(file_path)
    else: 
        df = DP.get_existing_csv(file_path)

    # Split into attacks-strings and labels
    X = DP.get_observed_data(df)
    y = DP.get_label(df)

    # Run performance evaluation for CountVectorizer
    #run_performance_evaluation('count', X, y)
    #run_performance_evaluation('tfid', X, y)
    #run_performance_evaluation('hash', X, y)

    #run_adversarial_attack(attack_type='PGD', sklearn_vectorizer='tfid', X=X, y=y, verbose=False)
    #run_adversarial_attack(attack_type='DT-Attack', sklearn_vectorizer='tfid', X=X, y=y, verbose=False)
    run_adversarial_attack(attack_type='SVM_Poisoning', sklearn_vectorizer='tfid', X=X, y=y, verbose=True)

def main():
    run_full_thesis_script()
if __name__ == "__main__":
    main()