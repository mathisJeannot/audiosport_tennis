import numpy as np
import matplotlib.pyplot as plt


def score_H(H:np.array, events_sec:np.array, threshold:float, segment_sec:float, Fs:int, prefixe_title='', win_length=1024, hop_length=512, bool_plot=False):
    """
    Algorithm for metric of all the rows of H
    ================================
    :param H: H to score
    :param events_sec: time-codes of the events
    :param threshold: threshold for detection (in [0,1])
    :param segment_sec: length of the segment (in sec) for the segment metric
    :param Fs: Sampling frequency for the stft
    :param prefixe_title: Prefixe for the title of the score result graph
    :param win_length: length of the window for stft
    :param hop_length: length of the hop for stft
    :param bool_plot: True to plot the score result graph and False otherwise
    --------------------------------
    :return: dict dict_score = {i:{'precision':p, 'recall':r, 'f_mesure':f}} where i are the index of the rows of H
    """
    dict_score = {i:{'precision':0, 'recall':0, 'f_mesure':0} for i in range(H.shape[0])}
    for i in range(H.shape[0]):
        precision, recall, f_mesure = score_composante(H[i], i, events_sec, threshold, segment_sec, Fs, prefixe_title, win_length, hop_length, bool_plot=bool_plot)
        dict_score[i]['precision']= precision
        dict_score[i]['recall']= recall
        dict_score[i]['f_mesure']= f_mesure
    return dict_score

def score_composante(h:np.array, index:int, events_sec:np.array, threshold:float, segment_sec:float, Fs:int, prefixe_title:str, win_length:int, hop_length:int, bool_plot:bool):
    """
    Algorithm for metric of a row of H
    ================================
    :param events_sec: time-codes of the events
    :param threshold: threshold for detection (in [0,1])
    :param segment_sec: length of the segment (in sec) for the segment metric
    :param Fs: Sampling frequency for the stft
    :param prefixe_title: Prefixe for the title of the score result graph
    :param win_length: length of the window for stft
    :param hop_length: length of the hop for stft
    :param bool_plot: True to plot the score result graph and False otherwise
    --------------------------------
    :return: float precision, float recal, float f_mesure
    """
    h_bin = h_to_h_bin(h, threshold, segment_sec, Fs, win_length, hop_length)
    events_bin = events_sec_to_events_bin(events_sec,  segment_sec, h_bin.shape[0])
    #computing of precision, recall and f_mesure    
    true_pos = sum(events_bin&h_bin)
    false_pos = sum(~events_bin&h_bin)
    false_neg = sum(events_bin&~h_bin) 
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f_mesure = 2*precision*recall/(precision+recall) if recall+precision!=0. else 0 #f_mesure = 0 if imposible to compute
    if bool_plot:
        plot_bin(index, h, h_bin, events_bin, threshold, precision, recall, f_mesure, prefixe_title)
    
    return precision, recall, f_mesure


def h_to_h_bin(h, threshold:float, segment_sec:float, Fs:int, win_length:int, hop_length:int):
    """
    Algorithm to get the binary activation of a row of h
    ================================
    :param h: a row of H
    :param threshold: threshold for detection (in [0,1])
    :param segment_sec: length of the segment (in sec) for the segment metric
    :param Fs: Sampling frequency for the stft
    :param win_length: length of the window for stft
    :param hop_length: length of the hop for stft
    --------------------------------
    :return: np.array h_bin (binary activation of h)
    """
    segment_len = sec_to_col(segment_sec, Fs, hop_length)
    nb_segments = int(h.shape[0]/segment_len)
    #normalisation de h
    max_h = max(h)
    h_norm = h/max_h
    #computing of binary h
    h_bin = np.zeros(nb_segments, dtype=int)
    for s in range(nb_segments):
        if s == nb_segments-1:
            for i in range(int(s*segment_len), h.shape[0]):
                if h_norm[i]>threshold:
                    h_bin[s]=1
                    break
        else:
            for i in range(int(s*segment_len), int((s+1)*segment_len)):
                if h_norm[i]>threshold:
                    h_bin[s]=1
                    break
    return h_bin
    
def events_sec_to_events_bin(events_sec:np.array,  segment_sec:float, nb_segments:int):
    """
    Algorithm to get the binary activation of the known events according to the segment length
    ================================
    :param events_sec: time-codes of the events
    :param segment_sec: length of the segment (in sec) for the segment metric
    :param nb_segments: total number of segments
    --------------------------------
    :return: np.array events_bin (binary activation of known events)
    """
    events_bin = np.zeros(nb_segments, dtype=int)
    for event_sec in events_sec:
        seg = int(event_sec/segment_sec)
        if seg<events_bin.shape[0]:
            events_bin[seg] = 1
    return events_bin
               
def plot_bin(index, h, h_bin, events_bin, threshold, precision, recall, f_mesure, prefixe_title=''):
    """
    Plot the score result of h. It allow to compare binary events with binary activation and activation
    ================================
    :param index: index (in H) of the row h
    :param h: a row of H
    :param event_bin: binary activation of known events
    :param threshold: threshold for detection (in [0,1])
    :param segment_sec: length of the segment (in sec) for the segment metric
    :param precision: precision of h_bin compared to events_bin
    :param recall: recall of h_bin compared to events_bin
    :param f_mesure: f mesure of h_bin compared to events_bin
    :param prefixe_title: prefixe for the title of the graph
    --------------------------------
    """
    max_h = max(h)
    h_norm = h/max_h
    fig, listeAxes = plt.subplots(3, 1, constrained_layout=True, figsize=(15,5))

    listeAxes[0].bar(range(events_bin.shape[0]),events_bin, color='m', label='Frappes de balle')
    listeAxes[0].legend()
    listeAxes[1].bar(range(h_bin.shape[0]),h_bin, label='Activation binaire')
    listeAxes[1].legend()
    listeAxes[2].plot(range(h_norm.shape[0]),h_norm, label='Activation')
    listeAxes[2].plot(range(h_norm.shape[0]),[threshold]*h.shape[0], color='r', label='Seuil')
    listeAxes[2].legend()
    listeAxes[0].set_title(prefixe_title + 'Composante ' +str(index) + '\n prec=' + str(round(precision,4)) + '   rec=' + str(round(recall,4)) + '    f_mes=' + str(round(f_mesure,4)))
    plt.show()
    

def plot_reglog(h, y_proba, y_bin, events_bin, threshold, precision, recall, f_mesure, prefixe_title=''):
    """
    Plot the score result of h. It allow to compare binary events with binary activation and activation
    ================================
    :param index: index (in H) of the row h
    :param h: the row of H corresponding to the event
    :param y_proba: activation of known events according to reggression logistique
    :param y_bin: binary activation of known events according to reglog
    :param event_bin: binary activation of known events
    :param threshold: threshold for detection (in [0,1])
    :param segment_sec: length of the segment (in sec) for the segment metric
    :param precision: precision of h_bin compared to events_bin
    :param recall: recall of h_bin compared to events_bin
    :param f_mesure: f mesure of h_bin compared to events_bin
    :param prefixe_title: prefixe for the title of the graph
    --------------------------------
    """
    max_h = max(h)
    h_norm = h/max_h
    fig, listeAxes = plt.subplots(4, 1, constrained_layout=True, figsize=(15,7))

    listeAxes[0].bar(range(events_bin.shape[0]),events_bin, color='m', label='Frappes de balle')
    listeAxes[0].legend()
    listeAxes[1].bar(range(y_bin.shape[0]),y_bin, label='Activation binaire')
    listeAxes[1].legend()
    listeAxes[2].plot(range(y_proba.shape[0]),y_proba, label='Activation')
    listeAxes[2].plot(range(y_proba.shape[0]),[threshold]*h.shape[0], color='r', label='Seuil')
    listeAxes[2].legend()
    listeAxes[3].plot(range(h_norm.shape[0]),h_norm, color='g', label='Activation de la NMF')
    listeAxes[3].legend()
    listeAxes[0].set_title(prefixe_title + 'prec=' + str(round(precision,4)) + '   rec=' + str(round(recall,4)) + '    f_mes=' + str(round(f_mesure,4)))
    plt.show()
    
def plot_activation(index, H, h_bin, events_bin, threshold, precision, recall, f_mesure, prefixe_title='', oth_ind=[]):
    """
    Plot the score result of h. It allow to compare binary events with binary activation and activation
    ================================
    :param index: index (in H) of the row h
    :param H: activation of a NMF
    :param h_bin: binary activation of the best row of H
    :param oth_ind: indexes of the activation to plot bellow the best activation
    :param event_bin: binary activation of known events
    :param threshold: threshold for detection (in [0,1])
    :param segment_sec: length of the segment (in sec) for the segment metric
    :param precision: precision of h_bin compared to events_bin
    :param recall: recall of h_bin compared to events_bin
    :param f_mesure: f mesure of h_bin compared to events_bin
    :param prefixe_title: prefixe for the title of the graph
    --------------------------------
    """
    max_h = max(h)
    h_norm = h/max_h
    fig, listeAxes = plt.subplots(3 + len(oth_ind), 1, constrained_layout=True, figsize=(15,5))

    listeAxes[0].bar(range(events_bin.shape[0]),events_bin, color='m', label='Frappes de balle')
    listeAxes[0].legend()
    listeAxes[0].set_title(prefixe_title + 'Composante ' +str(index) + '\n prec=' + str(round(precision,4)) + '   rec=' + str(round(recall,4)) + '    f_mes=' + str(round(f_mesure,4)))
    # Binary activation of the best component
    listeAxes[1].bar(range(h_bin.shape[0]),h_bin, label='Activation binaire')
    listeAxes[1].legend()
    #Normalised activation of the best component
    listeAxes[2].plot(range(h_norm.shape[0]),h_norm, label='Activation - ' + str(index))
    listeAxes[2].plot(range(h_norm.shape[0]),[threshold]*h.shape[0], color='r', label='Seuil')
    listeAxes[2].legend()
    #activation of the others components
    for i in range(len(oth_ind)):
        listeAxes[3+i].plot(range(h_norm.shape[0]),h_norm, label='Activation - ' + str(oth_ind[i]), color='g')
        listeAxes[3+i].legend()
    plt.show()
    
    
def sec_to_col(t_sec, Fs, hop_length):
    """
    Convert t_sec (secondes) to a number of column of the spectrogram
    ===============================
    :param t_sec: time to convert (in secondes)
    :param Fs: Sampling frequency of the stft
    :param hop_length: hop length for stft
    """
    return max(0, (t_sec*Fs)/hop_length)

def best_threshold(y_proba, h_event, event_sec, segment_sec, thresh_hop, Fs, win_length, hop_length, prefixe_title = '', plot_bool=True):
    """
    Algorithm to plot the roc curve and f_mesure in fonction of the threshold and compute the best threshold for postprocessing
    ================================
    :param y_proba: labels computed for event detection
    :param events_sec: time-codes of the events
    :param segment_sec: length of the segment (in sec) for the segment metric
    :param thresh_hop: hop between two consecutive tested threshold
    :param Fs: Sampling frequency for the stft
    :param win_length: length of the window for stft
    :param hop_length: length of the hop for stft
    --------------------------------
    :return: prec, rec, best_f_mes, best_thresh
    """
    precisions_rl, recalls_rl, f_mesures_rl, thresholds_rl  = [], [], [], []
    precisions_h, recalls_h, f_mesures_h, thresholds_h  = [], [], [], []
    for t in np.arange(0,1,thresh_hop):
        p_rl, r_rl, f_rl = score_composante(y_proba, 0, event_sec, t, segment_sec, Fs, None, win_length, hop_length, bool_plot=False)
        precisions_rl += [p_rl]
        recalls_rl += [r_rl] 
        f_mesures_rl += [f_rl]
        thresholds_rl += [t]
        p_h, r_h, f_h = score_composante(h_event, 0, event_sec, t, segment_sec, Fs, None, win_length, hop_length, bool_plot=False)
        precisions_h += [p_h]
        recalls_h += [r_h] 
        f_mesures_h += [f_h]
        thresholds_h += [t]
    if plot_bool:
        fig, listeAxes = plt.subplots(1, 2, constrained_layout=True, figsize=(8,4))
        listeAxes[0].plot(recalls_rl,precisions_rl, label='RegLog')
        listeAxes[0].plot(recalls_h,precisions_h, label='Activation NMF')
        listeAxes[0].legend()
        listeAxes[0].set_xlabel('Recall')
        listeAxes[0].set_ylabel('Precision')
        listeAxes[0].set_xlim(0,1)
        listeAxes[0].set_ylim(0,1)
        listeAxes[0].set_title(prefixe_title + 'Recall-Precision Curve')
        listeAxes[1].plot(thresholds_rl,f_mesures_rl, label='RegLog')
        listeAxes[1].plot(thresholds_h,f_mesures_h, label='Activation NMF')
        listeAxes[1].legend()
        listeAxes[1].set_xlabel('Threshold')
        listeAxes[1].set_ylabel('F mesure')
        listeAxes[1].set_xlim(0,1)
        listeAxes[1].set_ylim(0,1)
        listeAxes[1].set_title(prefixe_title + 'Threshold Curve')
        plt.show()
    best_f_mes_rl = max(f_mesures_rl)
    best_thresh_rl = thresholds_rl[f_mesures_rl.index(best_f_mes_rl)]
    prec_rl = precisions_rl[f_mesures_rl.index(best_f_mes_rl)]
    rec_rl = recalls_rl[f_mesures_rl.index(best_f_mes_rl)]
    return prec_rl, rec_rl, best_f_mes_rl, best_thresh_rl
