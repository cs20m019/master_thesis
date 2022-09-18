import random
import pandas as pd

normal_train_requests = '/datasets/normal_train_request.txt'
normal_test_requests = '/datasets/normal_test_request.txt'
anomalous_test_requests = '/datasets/anomalous_test_request.txt'

'''
dataframe helper methods
'''

def create_dataframe_from_sample(sample):
    df = pd.DataFrame(sample, columns=['request'])
    df['label'] = ' '
    return df

def assign_label_to_dfcolumn_label(dataframe, label):
    dataframe = dataframe.assign(label=label)
    return dataframe

def get_random_sample(filename, n_samples):
    import os
    current_dir = os.path.dirname(__file__)
    filename = current_dir + filename
    with open(filename) as f:
        r_sample = f.read().splitlines()
    r_sample = random.sample(r_sample, n_samples)
    return r_sample

def get_existing_csv(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

# split dataframe into columns
def get_observed_data(df):
    x = df.drop("label",axis = 1)
    return x

def get_label(df):
     y = df.label
     return y

'''
creates dataframe containing labeled CSIC 2010 normal and anomalous requests 
dataframe will consist of 60% normal and  40% anomalous requests
dataframe size is variable and randomly taken from the dataset files
'''
def get_csic2010_dataframe(dataframe_size):
    sample_size = int(dataframe_size)
    # create full dataframe, 60% normal traffic, 40% anomalous traffic
    df_normal = create_dataframe_from_sample(get_random_sample(normal_train_requests, (int(sample_size * 0.6))))
    df_anom = create_dataframe_from_sample(get_random_sample(anomalous_test_requests, (int(sample_size * 0.4))))
    df_normal = assign_label_to_dfcolumn_label(df_normal, 0)
    df_anom = assign_label_to_dfcolumn_label(df_anom, 1)
    df = pd.concat([df_normal, df_anom])
    return df

# Natural Language Processing 
''' Vectorize Globals
'''
min_df = 1
max_df = 1.0
max_features = 2000

def get_count_vectorizer(analyzer, ngram):
    from sklearn.feature_extraction.text import CountVectorizer
    count_vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df, max_df=max_df, max_features=max_features)
    return count_vectorizer

def get_hash_vectorizer(analyzer, ngram):
    from sklearn.feature_extraction.text import HashingVectorizer
    hash_vectorizer = HashingVectorizer(analyzer=analyzer,ngram_range=ngram, n_features=max_features)
    return hash_vectorizer

def get_tfidf_vectorizer(analyzer, ngram):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df, max_df=max_df, max_features=max_features)
    return tfidf_vectorizer

# Creates a sparse matrix from a document by providing different vectorize options of sklean
def vectorize_document(vec_option, analyzer, ngram, document):
    print('Creating sparse-matrix from document...')
    if vec_option == 'count':
        vectorizer = get_count_vectorizer(analyzer, ngram)
    elif vec_option == 'hash':
        vectorizer = get_hash_vectorizer(analyzer, ngram)
    elif vec_option == 'tfidf':
        vectorizer = get_tfidf_vectorizer(analyzer, ngram)
    else:
        print('Unsupported vectorize option: [', vec_option, ']')
        exit()
    sparse_matrix = vectorizer.fit_transform(document['request'])
    return sparse_matrix


'''
Optimization and analysis methods
'''
# Get a kfold cross-validator with split_number of folds
def get_crossvalidator_kfold(split_number):
    from sklearn.model_selection import KFold
    # Shuffle so both normal and anomalous records are in each fold
    kf = KFold(n_splits=split_number, shuffle=True, random_state=4)
    return kf

def max_abs_scale_data(sparse_matrix):
    # MaxAbsScaler is used for Scaling of sparse data
    from sklearn.preprocessing import MaxAbsScaler
    print('Scaling sparse-matrix with MaxAbsScaler...')
    scaler = MaxAbsScaler()
    # transform data
    scaled_sparse_matrix = scaler.fit_transform(sparse_matrix)
    return scaled_sparse_matrix

def normalize_data(sparse_matrix):
    # Normalize sparse matrix
    from sklearn.preprocessing import normalize
    print('Normalizing sparse-matrix...')
    normalized_matrix  = normalize(sparse_matrix)
    return normalized_matrix
