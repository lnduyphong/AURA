import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

device = "cuda" if torch.cuda.is_available else "cpu"
print("Device:", device)

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, num_classes, alpha=0.1, q=0.7):
    # Convert logits to probabilities
    probs_1 = F.softmax(y_1, dim=1)
    probs_2 = F.softmax(y_2, dim=1)
    
    # Label Smoothing
    smoothed_targets = (1 - alpha) * F.one_hot(t, num_classes) + alpha/num_classes
    
    # Symmetric Cross Entropy (SCE)
    ce_1 = F.cross_entropy(y_1, t, reduction='none')
    rce_1 = -torch.sum(smoothed_targets * F.log_softmax(y_1, dim=1), dim=1)
    loss_1 = ce_1 + 0.1 * rce_1  # SCE with α=1, β=0.1

    # Generalized Cross Entropy (GCE)
    loss_2 = (1 - torch.pow(probs_2[torch.arange(probs_2.size(0)), t], q)) / q
    
    # Combine both losses
    combined_loss_1 = 0.6 * loss_1 + 0.4 * loss_2
    combined_loss_2 = 0.6 * loss_2 + 0.4 * loss_1
    
    # Co-teaching sample selection
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(t))
    
    # Sort losses
    ind_1_sorted = torch.argsort(combined_loss_1)
    ind_2_sorted = torch.argsort(combined_loss_2)
    
    # Select clean samples
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    
    # Final loss calculation with Bootstrapping
    final_loss_1 = F.cross_entropy(y_1[ind_2_update], t[ind_2_update]) + \
                  0.2 * F.kl_div(F.log_softmax(y_1[ind_2_update], dim=1),
                                probs_2[ind_2_update].detach(), reduction='batchmean')
    
    final_loss_2 = F.cross_entropy(y_2[ind_1_update], t[ind_1_update]) + \
                  0.2 * F.kl_div(F.log_softmax(y_2[ind_1_update], dim=1),
                                probs_1[ind_1_update].detach(), reduction='batchmean')
    
    return final_loss_1, final_loss_2

