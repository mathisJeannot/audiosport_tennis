import numpy as np
import soundfile as sf
import rebuild_signal as rs
import algos_nmfs as an
import os


def execute_and_export_all_NMFS_N_best(x, X, rank_list:list, beta_list:list, tau_list:list, N, Fs, t_codes, path:str, win_length=1024, hop_length=512, nmf=True, snmf=True, cnmf=True):
    """
    Execute differents NMF (CNMF, SNMF and NMF) with different ranks. Export the signal of all the components of an NMF, and the signals composed of the n best components on the zone ([beg1, end1], [beg2, end2], ...), for n in {1, ..., N}. And export them in path.
    ================================
    :param x: the original signal
    :param X: the complexe STFT of x
    :param rank_list: list of the ranks to applie to the several NMFs
    :param beta_list: list of beta for the beta-divergence
    :param tau_list: list of tau for the CNMF
    :param N: max number of templates used (if n=0 all the templates are used)
    :param Fs: sampling frequency
    :param t_codes: table of tuble [(beg,end), ...] where beg is the begining (in seconde) of an event and end the end of this event.
    :param path: path of the export
    :param win_length: 
    :param hop_length:
    :param nmf: True to compute the NMF, False if not
    :param snmf: True to compute the SNMF, False if not
    :param cnmf: True to compute the CNMF, False if not
    --------------------------------
    :return: x_hat, matrix H
    """
    if nmf:
        execute_and_export_NMF_N_best(x, X, rank_list, beta_list, N, Fs, t_codes, path, win_length, hop_length)
    if snmf:
        execute_and_export_SNMF_N_best(x, X, rank_list, beta_list, N, Fs, t_codes, path, win_length, hop_length)
    if cnmf:
        execute_and_export_CNMF_N_best(x, X, rank_list, beta_list, tau_list, N, Fs, t_codes, path, win_length, hop_length)


#CNMF
def execute_and_export_CNMF_N_best(x, X, rank_list:list, beta_list:list, tau_list:list, N, Fs, t_codes, path:str, win_length=1024, hop_length=512):
    """
    Execute CNMF with different ranks, tau and divergences. Export the signal of all the components of an NMF, and the signals composed of the n best components on the zone ([beg1, end1], [beg2, end2], ...), for n in {1, ..., N}. And export them in path.
    ================================
    :param x: the original signal
    :param X: the complexe STFT of x
    :param rank_list: list of the ranks to applie to the several NMFs
    :param beta_list: list of beta for the beta-divergence
    :param tau_list: list of tau for the CNMF
    :param N: max number of templates used (if n=0 all the templates are used)
    :param Fs: sampling frequency
    :param t_codes: table of tuble [(beg,end), ...] where beg is the begining (in seconde) of an event and end the end of this event.
    :param path: path of the export
    :param win_length: 
    :param hop_length: 
    --------------------------------
    :return: x_hat, matrix H
    """
    V = abs(X)**2
    mkfold(path)
    n_cnmf_tot = len(rank_list)*len(beta_list)*len(tau_list)
    n_cnmf_completed = 0
    for rank in rank_list:
        print('-----Begining rank ', rank)
        path_rank = mkfold(path + '/NMFs_de_rang_' + str(rank) + '/')
        for beta in beta_list:
            path_beta = mkfold(path_rank + '/beta_divergence_' + str(beta) +'/')
            path_CNMF = mkfold(path_beta + '/CNMF/')
            for tau in tau_list:
                path_CNMF_tau = mkfold(path_CNMF + '/tau_' + str(tau) + '/')
                n_cnmf_completed+=1
                print('CNMF ' +  str(n_cnmf_completed) + '/' + str(n_cnmf_tot) + ' in progress...')
                
                W_CNMF, H_CNMF = an.CNMF(V, rank, tau, beta=beta)
                export_NMF_1_to_N_best(x, X, W_CNMF, H_CNMF, N, Fs, t_codes, path_CNMF_tau, win_length, hop_length)
                print('CNMF completed succesfully')

#NMF
def execute_and_export_NMF_N_best(x, X, rank_list:list, beta_list:list, N, Fs, t_codes, path:str, win_length=1024, hop_length=512):
    """
    Execute NMF with different ranks, tau and divergences. Export the signal of all the components of an NMF, and the signals composed of the n best components on the zone ([beg1, end1], [beg2, end2], ...), for n in {1, ..., N}. And export them in path.
    ================================
    :param x: the original signal
    :param X: the complexe STFT of x
    :param rank_list: list of the ranks to applie to the several NMFs
    :param beta_list: list of beta for the beta-divergence
    :param N: max number of templates used (if n=0 all the templates are used)
    :param Fs: sampling frequency
    :param t_codes: table of tuble [(beg,end), ...] where beg is the begining (in seconde) of an event and end the end of this event.
    :param path: path of the export
    :param win_length: 
    :param hop_length: 
    --------------------------------
    :return: x_hat, matrix H
    """
    V = abs(X)**2
    mkfold(path)
    n_nmf_tot = len(rank_list)*len(beta_list)
    n_nmf_completed = 0
    for rank in rank_list:
        print('-----Begining rank ', rank)
        path_rank = mkfold(path + '/NMFs_de_rang_' + str(rank) + '/')
        for beta in beta_list:
            path_beta = mkfold(path_rank + '/beta_divergence_' + str(beta) +'/')
            n_nmf_completed+=1
            print('NMF ' +  str(n_nmf_completed) + '/' + str(n_nmf_tot) + ' in progress...')
            path_NMF = mkfold(path_beta + '/NMF/')
            W_NMF, H_NMF = an.NMF(V, rank, beta=beta)
            export_NMF_1_to_N_best(x, X, W_NMF, H_NMF, N, Fs, t_codes, path_NMF, win_length, hop_length)
            print('NMF completed succesfully')
        

#SNMF
def execute_and_export_SNMF_N_best(x, X, rank_list:list, beta_list:list, N, Fs, t_codes, path:str, win_length=1024, hop_length=512):
    """
    Execute NMF with different ranks, tau and divergences. Export the signal of all the components of an NMF, and the signals composed of the n best components on the zone ([beg1, end1], [beg2, end2], ...), for n in {1, ..., N}. And export them in path.
    ================================
    :param x: the original signal
    :param X: the complexe STFT of x
    :param rank_list: list of the ranks to applie to the several NMFs
    :param beta_list: list of beta for the beta-divergence
    :param N: max number of templates used (if n=0 all the templates are used)
    :param Fs: sampling frequency
    :param t_codes: table of tuble [(beg,end), ...] where beg is the begining (in seconde) of an event and end the end of this event.
    :param path: path of the export
    :param win_length: 
    :param hop_length: 
    --------------------------------
    :return: x_hat, matrix H
    """
    V = abs(X)**2
    mkfold(path)
    n_snmf_tot = len(rank_list)*len(beta_list)
    n_snmf_completed = 0
    for rank in rank_list:
        print('-----Begining rank ', rank)
        path_rank = mkfold(path + '/NMFs_de_rang_' + str(rank) + '/')
        for beta in beta_list:
            path_beta = mkfold(path_rank + '/beta_divergence_' + str(beta) +'/')
            n_snmf_completed+=1
            print('SNMF ' +  str(n_snmf_completed) + '/' + str(n_snmf_tot) + ' in progress...')
            path_SNMF = mkfold(path_beta + '/SNMF/')
            W_SNMF, H_SNMF = an.SNMF(V, rank)
            export_NMF_1_to_N_best(x, X, W_SNMF, H_SNMF, N, Fs, t_codes, path_SNMF, win_length, hop_length)
            print('SNMF completed succesfully')        
        
"""
        #NMF
        print('NMF in progress...')
        path_NMF = mkfold(path_rank + '/NMF/')
        W_NMF, H_NMF = an.NMF(V, rank)
        export_NMF_1_to_N_best(x, X, W_NMF, H_NMF, N, Fs, t_codes, path_NMF, win_length, hop_length)
        print('NMF completed succesfully')
        
        #SNMF
        print('SNMF in progress...')
        path_SNMF = mkfold(path_rank + '/SNMF/')
        W_SNMF, H_SNMF = an.SNMF(V, rank)
        export_NMF_1_to_N_best(x, X, W_SNMF, H_SNMF, N, Fs, t_codes, path_SNMF, win_length, hop_length)
        print('SNMF completed succesfully')
        
        #CNMF
        print('CNMF in progress...')
        path_CNMF = mkfold(path_rank + '/CNMF/')
        W_CNMF, H_CNMF = an.CNMF(V, rank, tau)
        export_NMF_1_to_N_best(x, X, W_CNMF, H_CNMF, N, Fs, t_codes, path_CNMF, win_length, hop_length)
        print('CNMF completed succesfully')
"""
def mkfold(path):
    try:
        os.mkdir(path)
    except:
        None
    return path
 
    
            
def export_NMF_1_to_N_best(x, X, W, H, N, Fs, t_codes:np.array, path:str, win_length=1024, hop_length=512):
    """
    export as .wav files the signal of all the components of an NMF, and the signals composed of the n best components on the zone ([beg1, end1], [beg2, end2], ...), for n in {1, ..., N}
    
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
    x_hat = rs.rebuild_components_NMF(x, X, W, H, win_length, hop_length)
    sorted_list = rs.sort_max_H_time_codes(H, Fs, t_codes, win_length, hop_length)
    
    #creation of the folders
    path_comp_wise = path + '/component_wise'
    path_1_to_N_bests = path + '/1_to_' + str(N) + '_bests_components'
    try:
        os.mkdir(path_comp_wise)
    except:
        None
    try:
        os.mkdir(path_1_to_N_bests)
    except:
        None
    
    #filling of the arrays of signals
    x_hat_N = np.zeros((N+1,x.shape[0]))
    for n in range(1, N+1):
        x_hat_N[n] = x_hat_N[n-1] + x_hat[:, int(sorted_list[n-1][1])] #add the n-th best components
            
    #Export of the signals
    for i in range(rank): #component wise
        path_i = path_comp_wise + '/component_' + str(i+1) + '.wav'
        sf.write(path_i, x_hat[:,i], Fs)
    for n in range(1,N+1): #compound signals
        components = str(int(sorted_list[0][1])+1)
        for k in range(1,n):
            components += '-' + str(int(sorted_list[k][1])+1)
        path_n = path_1_to_N_bests + '/components_' + components + '.wav'
        sf.write(path_n, x_hat_N[n], Fs)
        
        
        