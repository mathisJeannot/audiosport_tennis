import os, sys
import numpy as np
import matplotlib.pyplot as plt

def power_to_db(V, amin=1e-10, top_db=80.0):
    """
    Taken from https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    
    Essentially computes a power spectrogram in dB.
    """
    
    ref = np.max(V)
    V_dB = 10.0 * np.log10(np.maximum(amin, V))
    V_dB -= 10.0 * np.log10(np.maximum(amin, ref))
    V_dB = np.maximum(V_dB, V_dB.max() - top_db)
    return V_dB

def amp_to_db(V, amin=1e-10, top_db=80.0):
    """
    Taken from https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    
    Essentially computes a power spectrogram in dB.
    """
    return power_to_db(V**2, amin, top_db)


def plot_NMF(V, W, H, figsize=(15,3), aspect='auto', wr=[1, 0.5, 1, 1], power_spec=True):
    """
    Plot the NMF of V
    ================================
    :param V: original data matrix
    :param W: dictionary matrix
    :param H: activation matrix
    :param figsize: size of the figure
    :param aspect: controls the aspect ratio oof the axes
    :param wr: width ratios of the columns
    :param power_spec: set to True if V is a power spectrogramme
    """

    if power_spec:
        V = power_to_db(V)
        V_hat = power_to_db(W@H)
        W = power_to_db(W)
        H = np.sqrt(np.sqrt(H))

    fig, ax = plt.subplots(1, 4, gridspec_kw={'width_ratios': wr}, figsize=figsize)

    rank = W.shape[1]
    cmap = 'gray_r'
    #Plot V
    im = ax[0].imshow(V, aspect=aspect, origin='lower', cmap=cmap)
    ax[0].set_title(r'$V$')
    plt.sca(ax[0])
    plt.colorbar(im)
    #Plot W
    im = ax[1].imshow(W, aspect=aspect, origin='lower', cmap=cmap)
    ax[1].set_title(r'$W$')
    plt.sca(ax[1])
    plt.colorbar(im)
    #plt.xticks(np.arange(K), np.arange(1, K+1))
    #Plot H
    im = ax[2].imshow(np.flip(H, axis=0), aspect=aspect, origin='lower', cmap=cmap)
    ax[2].set_title(r'$H$')
    plt.sca(ax[2])    
    plt.colorbar(im)
    #plt.yticks(np.arange(K), np.arange(K, 0, -1))
    #plot V_hat
    im = ax[3].imshow(V_hat, aspect=aspect, origin='lower', cmap=cmap)
    ax[3].set_title(r'$WH$')
    plt.sca(ax[3])    
    plt.colorbar(im)
    plt.tight_layout() 
    plt.show() 

def plot_spectogramm(V, Fs, win_length=1024, hop_length=512):
    freq = np.arange(0, V.shape[0])*Fs/win_length 
    frames = np.arange(0, V.shape[1])*(hop_length/Fs)

    plt.figure(figsize=(10,7))
    plt.imshow(power_to_db(V), aspect='auto', origin='lower', cmap='gray_r')
    plt.yticks(np.arange(0, V.shape[0], 100), np.round(freq[0:-1:100]).astype(int))
    plt.xticks(np.arange(0, V.shape[1], 100), np.round(frames[0:-1:100]).astype(int))
    plt.colorbar()   
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    plt.title('power spectrogram')
    plt.show()

def plot_H(H, figsize=(10,10)):
    rank = H.shape[0]
    H = np.sqrt(np.sqrt(H))
    for i in range(rank):
        plt.subplot(rank, 1, i+1)
        plt.plot(H[i])
    plt.show()
           
