import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import convolutive_MM





def rebuild_signal_n_best(x, X, W, H, n, Fs, t_codes, win_length=1024, hop_length=512):
    """
    Rebuild signal algorithm for NMF with the n best templates on the zone [beg, end].
    ================================
    :param x: the original signal
    :param X: the complexe STFT of x
    :param W: the dictionnary matrix
    :param H: the activation matrix
    :param n: number of templates used (if n=0 all the templates are used)
    :param Fs: sampling frequency
    :param beg: time-code of the begining of the time zone
    :param win: length of the time zone
    :param win_length: 
    :param hop_length: 
    --------------------------------
    :return: matrix W, matrix H
    """
    x_componed = np.zeros(x.shape)
    if n:
        sorted_list = sort_max_H_time_codes(H, Fs, t_codes, win_length, hop_length)
        x_hat = rebuild_n_components_NMF_sorted_list(x, X, W, H, n, sorted_list, win_length, hop_length)  
        for k in range( min(n, len(sorted_list)) ):
            x_componed += x_hat[:,int(sorted_list[k][1])]
    else:
        x_hat = rebuild_signal_NMF(x, X, W, H, win_length, hop_length)
        for k in range(x_hat.shape[1]):
            x_componed += x_hat[:,k]
    return x_componed


def rebuild_components_NMF(x, X, W, H, win_length=1024, hop_length=512):
    rank = H.shape[0]
    length = x.shape[0]
    x_hat = np.zeros((length, rank))

    for k in range(rank):
        if len(W.shape)==2: #NMF
            V_hat_k = W[:, k, np.newaxis]@H[np.newaxis, k, :]
            
        elif len(W.shape)==3:# CNMF
            V_hat_k = sum(np.dot(W[t][:, k, np.newaxis], convolutive_MM.shift(H,t)[np.newaxis, k, :]) for t in range(W.shape[0]))
        X_hat_k = np.sqrt(V_hat_k)*np.exp(1j*np.angle(X))
        x_hat_k = librosa.istft(X_hat_k, hop_length=hop_length, win_length=win_length,
                           window='hann', center = True, length=length)
        x_hat[:,k] = x_hat_k
    return x_hat

def rebuild_n_components_NMF_sorted_list(x, X, W, H, n, sorted_list, win_length=1024, hop_length=512):
    rank = H.shape[0]
    length = x.shape[0]
    x_hat = np.zeros((length, rank))

    for i in range(min(n, len(sorted_list))):
        k = int(sorted_list[i][1])          
        if len(W.shape)==2: #NMF
            V_hat_k = W[:, k, np.newaxis]@H[np.newaxis, k, :]
            
        elif len(W.shape)==3: #CNMF
            V_hat_k = sum(np.dot(W[t][:, k, np.newaxis], convolutive_MM.shift(H,t)[np.newaxis, k, :]) for t in range(W.shape[0]))
        X_hat_k = np.sqrt(V_hat_k)*np.exp(1j*np.angle(X))
        x_hat_k = librosa.istft(X_hat_k, hop_length=hop_length, win_length=win_length,
                           window='hann', center = True, length=length)
        x_hat[:,k] = x_hat_k
    return x_hat




def sort_max_H_beg_end(H, Fs, beg, end, win_length, hop_length):
    """
    Trie de manière décroissante les sous-bande de H,
    selon la somme des coefficient entre beg et end (en seconde)
    """
    sorted_list = []
    col_beg = int((beg*Fs-win_length)/hop_length)
    col_end = int((end*Fs-win_length)/hop_length)
    for i in range(H.shape[0]):
        sorted_list += [[np.sum(H[i, col_beg:col_end]), i]]
    return np.array(sorted(sorted_list, key=lambda x: x[0], reverse=True))

def sort_max_H_time_codes(H, Fs, t_codes:np.array, win_length, hop_length):
    """
    Trie de manière décroissante les sous-bande de H,
    selon la somme des coefficient entre beg et end (en seconde)
    """
    sorted_list = []
    col_H_indices = ((t_codes*Fs-win_length)/hop_length).astype(int) #convertit le tableau des times codes en indices de colonnes de H
    for i in range(H.shape[0]):
        total_i = 0
        for w in col_H_indices:
            total_i += np.sum(H[i, w[0]:w[1]])
        sorted_list += [[total_i, i]]
    return np.array(sorted(sorted_list, key=lambda x: x[0], reverse=True))



def rebuild_and_tidy_NMF_to_N_best(x, X, W, H, N, Fs, t_codes:np.array, path:str, win_length=1024, hop_length=512):
    """
    return the signal of all the components of an NMF, and the signals composed of the n best components on the zone ([beg1, end1], [beg2, end2], ...), for n in {1, ..., N}. And export them in path.
    
    ================================
    :param x: the original signal
    :param X: the complexe STFT of x
    :param W: the dictionnary matrix
    :param H: the activation matrix
    :param N: max number of templates used (if n=0 all the templates are used)
    :param Fs: sampling frequency
    :param t_codes: table of tuble [(beg,end), ...] where beg is the begining (in seconde) of an event and end the end of this event.
    :param path: path of the export
    :param win_length: 
    :param hop_length: 
    --------------------------------
    :return: x_hat, matrix H
    """
    rank = H.shape[0]
    N = min(rank, N)
    x_hat = rebuild_components_NMF(x, X, W, H, win_length, hop_length)
    sorted_list = sort_max_H_time_codes(H, Fs, t_codes, win_length, hop_length)
    
    #filling of the arrays of signals
    x_hat_N = np.array((N+1,x.shape[0]))
    for n in range(1, N+1):
        if n:
            x_hat_N[n] = x_hat_N[n-1] + x_hat[:, sorted_list[n][1]] #add the n-th best components
            
    #Export of the signals
    for i in range(rank):
        path_i = path + '/component_by_component/component_' + str(i) + '.wav'
        sf.write(path_i, x_hat[i], Fs)
    for n in range(1,N+1):
        components = str(sorted_list[0][1])
        for k in range(1,n):
            components += '-' + str(sorted_list[k][1])
        path_n = path + '/1_to_' + str(N) + '_bests_components/components_' + components + '.wav'
        
    
    
def plot_signal(x, Fs, figsize=(15,2)):
    time = np.arange(0,x.shape[0]/Fs, 1/Fs)
    
    plt.figure(figsize=figsize)
    plt.plot(time, x, 'k') # plot this component
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('waveform')
    plt.ylim([-1,1])
    plt.show()
