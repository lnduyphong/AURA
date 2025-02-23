import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

device = "cuda" if torch.cuda.is_available else "cpu"
print("Device:", device)

# Loss functions
def loss_coteaching(probs_1, probs_2, probs_3, targets, forget_rate, num_classes, alpha=0.1, q=0.7):
    # Label Smoothing
    smoothed_targets = (1 - alpha) * F.one_hot(targets, num_classes) + alpha/num_classes
    

    loss_1 = (1 - torch.pow(probs_1[torch.arange(probs_1.size(0)), targets], q)) / q

    # Generalized Cross Entropy (GCE)
    loss_2 = (1 - torch.pow(probs_2[torch.arange(probs_2.size(0)), targets], q)) / q
    
    # Kullback-Leibler Divergence (KLD)
    loss_3 = (1 - torch.pow(probs_2[torch.arange(probs_2.size(0)), targets], q)) / q
    
    # Combine both losses
    combined_loss_32 = 0.45 * loss_3 + 0.45 * loss_2 + 0.1 * loss_1
    combined_loss_13 = 0.45 * loss_1 + 0.45 * loss_3 + 0.1 * loss_2
    combined_loss_21 = 0.45 * loss_2 + 0.45 * loss_1 + 0.1 * loss_3
    
    # Co-teaching sample selection
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(targets))
    
    # Sort losses
    ind_1_sorted = torch.argsort(combined_loss_32)
    ind_2_sorted = torch.argsort(combined_loss_13)
    ind_3_sorted = torch.argsort(combined_loss_21)
    
    # Select clean samples
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    ind_3_update = ind_3_sorted[:num_remember]
    
    final_loss_1 = F.cross_entropy(probs_1[ind_1_update], targets[ind_1_update]) + \
                  0.2 * F.kl_div(F.log_softmax(probs_1[ind_1_update], dim=1),
                                probs_2[ind_1_update].detach(), reduction='batchmean')
    
    final_loss_2 = F.cross_entropy(probs_2[ind_2_update], targets[ind_2_update]) + \
                  0.2 * F.kl_div(F.log_softmax(probs_2[ind_2_update], dim=1),
                                probs_3[ind_2_update].detach(), reduction='batchmean')
                  
    final_loss_3 = F.cross_entropy(probs_3[ind_3_update], targets[ind_3_update]) + \
                  0.2 * F.kl_div(F.log_softmax(probs_3[ind_3_update], dim=1),
                                probs_1[ind_3_update].detach(), reduction='batchmean')
    
    return final_loss_1, final_loss_2, final_loss_3

