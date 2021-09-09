my_seed = 1
import numpy as np
np.random.seed(my_seed)
import torch
torch.manual_seed(my_seed)
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import convertion as conv



def y_pred_proba_overlap(H_train, H_val, event_sec_train, event_sec_val, half_win, Fs, hop_length, class_weight=None):
    """
    Compute the probabilities for beeing of class 1
    ================================
    :param H: H of an NMF
    --------------------------------
    :return y_proba: probabilities for the event in the test data
    """
    #Preparation of the data
    X_train = X_from_H(H_train, half_win)
    X_val = X_from_H(H_val, half_win)
    Y_train = np.zeros(X_train.shape[0])
    for t in event_sec_train:
        Y_train[conv.sec_to_col(t, Fs, hop_length)]=1
    Y_val = np.zeros(X_val.shape[0])
    for t in event_sec_val:
        Y_val[conv.sec_to_col(t, Fs, hop_length)]=1
    
    X_train_tensor = torch.from_numpy(X_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    Y_train_tensor = torch.from_numpy(Y_train).float()
    Y_val_tensor = torch.from_numpy(Y_val).float()
    
    input_dim = X_train.shape[1]
    
    #Neural Network
    my_nn = MyFirstMLP(input_dim=input_dim, output_dim=1, hidden_dim=50)
    
    #train mode
    my_nn.train()
    
    #hyperparameters
    learning_rate = 1
    num_epochs = 200
    
    #optimizer
    #optimizer = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)
    optimizer = torch.optim.LBFGS(my_nn.parameters(), lr=learning_rate, max_iter=100)
    
    #Weight - train
    n_samples_train = int(Y_train_tensor.size()[0])
    n1_train = int(torch.sum(Y_train_tensor))
    n0_train = n_samples_train - n1_train
    w0_train= n_samples_train/(2*n0_train)
    w1_train = n_samples_train/(2*n1_train)
    weight_train = Y_train_tensor*w1_train + (1-Y_train_tensor)*w0_train
    #Weight - val
    n_samples_val = int(Y_val_tensor.size()[0])
    n1_val = int(torch.sum(Y_val_tensor))
    n0_val = n_samples_val - n1_val
    w0_val= n_samples_val/(2*n0_val)
    w1_val = n_samples_val/(2*n1_val)
    weight_val = Y_val_tensor*w1_val + (1-Y_val_tensor)*w0_val
    
    #the Loss
    loss_fn = nn.BCELoss(reduction='none')
    
    train_loss = []
    val_loss = []
    Y_prob_train = []
    def closure():
        # reset the stored gradients for the parameters of the neural network
        my_nn.zero_grad()    
        # do the forward pass
        Y_hat = my_nn(X_train_tensor)    # compute and store the loss on the training set
        Y_prob_train = Y_hat
        loss = loss_fn(Y_hat, Y_train_tensor.unsqueeze(1)) # compute the loss
        loss = loss*weight_train
        loss = torch.mean(loss)
        train_loss.append(loss.item()) # store the loss
        # do the backward pass
        loss.backward()
        return loss

    for epoch in range(num_epochs):
        """#reset the stored gradient
        my_nn.zero_grad()
        
        Y_prob_train = my_nn(X_train_tensor)
        loss = loss_fn(Y_prob_train, Y_train_tensor.unsqueeze(1))
        loss = loss*weight_train
        loss = torch.mean(loss)
        train_loss.append(loss.item())
        
        loss.backward()"""
        
    
        #gradient descent step
        optimizer.step(closure)
        
        Y_prob_val = my_nn(X_val_tensor)
        loss = loss_fn(Y_prob_val, Y_val_tensor.unsqueeze(1))
        loss = loss*weight_train
        loss = torch.mean(loss)
        val_loss.append(loss.item())

    
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(["training", "validation"])
    plt.title("loss")
    plt.xlabel("epochs")
    plt.show()
    
    return Y_prob_train.detach().numpy()[:,0], Y_prob_val.detach().numpy()[:,0]




#Definition of a nn for binary classification
class MyFirstMLP(nn.Module):

    def __init__(self, input_dim=2, output_dim=1, hidden_dim=50):
        
        super(MyFirstMLP, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        """
        self.layer1 = nn.Linear(input_dim, hidden_dim) 
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()"""
        
        
    def forward(self, x):
        
        x = self.layer1(x) # x <- A1 x + b1
        #x = self.tanh(x) # x <- tanh(x)
        #x = self.layer2(x)  # x <- A2 x + b2
        x = self.sigmoid(x) # x <- sigmoid(x)
        
        return x
    
    
    

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
                
            