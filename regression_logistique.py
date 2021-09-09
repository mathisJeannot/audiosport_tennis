from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import convertion as conv
import numpy as np

def y_pred_proba(H_train, H_test, event_sec_train, event_sec_test, Fs, hop_length, verbose=1):
    """
    Compute the probabilities for beeing of class 1
    ================================
    :param H: H of an NMF
    --------------------------------
    :return y_proba: probabilities for the event in the test data
    """
    #Preparation of the data
    X_train = H_train.T
    X_test = H_test.T
    y_train = np.zeros(X_train.shape[0])
    for t in event_sec_train:
        y_train[conv.sec_to_col(t, Fs, hop_length)]=1
    y_test = np.zeros(X_test.shape[0])
    for t in event_sec_train:
        y_train[conv.sec_to_col(t, Fs, hop_length)]=1
    
    #Normalisation   => Centrer et réduire pb car pas positif ????????????????????
    sc= StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test) 
    
    #Construction of the model
    classifier = LogisticRegression(random_state=0, solver="liblinear", verbose=verbose)
    classifier.fit(X_train, y_train)
    
    #Prediction
    y_prob = classifier.predict_proba(X_test)
    
    return y_prob[:,1]


def y_pred_proba_overlap(H_train, H_test, event_sec_train, event_sec_test, half_win, Fs, hop_length, class_weight=None):
    """
    Compute the probabilities for beeing of class 1
    ================================
    :param H: H of an NMF
    --------------------------------
    :return y_proba: probabilities for the event in the test data
    """
    #Preparation of the data
    X_train = X_from_H(H_train, half_win)
    X_test = X_from_H(H_test, half_win)
    y_train = np.zeros(X_train.shape[0])
    for t in event_sec_train:
        y_train[conv.sec_to_col(t, Fs, hop_length)]=1
    y_test = np.zeros(X_test.shape[0])
    for t in event_sec_test:
        y_test[conv.sec_to_col(t, Fs, hop_length)]=1
    
    #Normalisation   => Centrer et réduire pb car pas positif ????????????????????
    sc= StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test) 
    
    #Construction of the model
    classifier = LogisticRegression(random_state=0, solver="liblinear", verbose=1, class_weight=class_weight)
    classifier.fit(X_train, y_train)
    
    #Prediction
    y_prob_train = classifier.predict_proba(X_train)
    y_prob_test = classifier.predict_proba(X_test)
    coef = classifier.coef_
    
    return y_prob_train[:,1], y_prob_test[:,1], coef


def y_pred_proba_overlap_mean_std(H_train, H_test, event_sec_train, event_sec_test, half_win, Fs, hop_length, class_weight=None):
    """
    Compute the probabilities for beeing of class 1
    ================================
    :param H: H of an NMF
    --------------------------------
    :return y_proba: probabilities for the event in the test data
    """
    #Preparation of the data
    X_train = X_from_H_mean_std(H_train, half_win)
    X_test = X_from_H_mean_std(H_test, half_win)
    y_train = np.zeros(X_train.shape[0])
    for t in event_sec_train:
        y_train[conv.sec_to_col(t, Fs, hop_length)]=1
    y_test = np.zeros(X_test.shape[0])
    for t in event_sec_test:
        y_test[conv.sec_to_col(t, Fs, hop_length)]=1
    
    #Normalisation   => Centrer et réduire pb car pas positif ????????????????????
    sc= StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test) 
    
    #Construction of the model
    classifier = LogisticRegression(random_state=0, solver="liblinear", verbose=1, class_weight=class_weight)
    classifier.fit(X_train, y_train)
    
    #Prediction
    y_prob_train = classifier.predict_proba(X_train)
    y_prob_test = classifier.predict_proba(X_test)
    coef = classifier.coef_
    
    return y_prob_train[:,1], y_prob_test[:,1], coef



def X_from_H(H, half_win):
    nb_col=H.shape[1]
    X = []
    for i in range(nb_col):
        
        #Left side effect
        if i-half_win<0:
            concat_list = []
            nb_out = half_win - i #Number of column out of range
            for j in range(nb_out):
                concat_list += [H[:, 0]]
            for j in range(2*half_win+1-nb_out):
                concat_list += [H[:, j]]
            X += [np.concatenate(concat_list)]
        
        #Rigth side effect
        elif i+half_win>=nb_col:
            concat_list = []
            nb_out = i + half_win - nb_col +1 #Number of column out of range
            for j in range(2*half_win+1-nb_out):
                concat_list += [H[:, i-half_win+j]]
            for j in range(nb_out):
                concat_list += [H[:, nb_col-1]]
            X += [np.concatenate(concat_list)]
        
        #Common case
        else:
            concat_list = []
            for j in range(0, half_win+1):
                concat_list += [H[:, i-j]]
            for j in range(1, half_win+1):
                concat_list += [H[:, i+j]]
            X += [np.concatenate(concat_list)]
            
    return np.array(X)                     
                
def X_from_H_mean_std(H, half_win):
    nb_col=H.shape[1]
    X = []
    for i in range(nb_col):
        
        #Left side effect
        if i-half_win<0:
            concat_list = []
            nb_out = half_win - i #Number of column out of range
            for j in range(2*half_win+1-nb_out):
                concat_list += [H[:, j]]
            col = np.array(concat_list)
            mean = np.mean(col, axis=0)
            std = np.std(col, axis=0)
            X += [np.concatenate([mean, std])]
        
        #Rigth side effect
        elif i+half_win>=nb_col:
            concat_list = []
            nb_out = i + half_win - nb_col +1 #Number of column out of range
            for j in range(2*half_win+1-nb_out):
                concat_list += [H[:, i-half_win+j]]
            col = np.array(concat_list)
            mean = np.mean(col, axis=0)
            std = np.std(col, axis=0)
            X += [np.concatenate([mean, std])]
        
        #Common case
        else:
            concat_list = []
            for j in range(0, half_win+1):
                concat_list += [H[:, i-j]]
            for j in range(1, half_win+1):
                concat_list += [H[:, i+j]]
            col = np.array(concat_list)
            mean = np.mean(col, axis=0)
            std = np.std(col, axis=0)
            X += [np.concatenate([mean, std])]
            
    return np.array(X) 