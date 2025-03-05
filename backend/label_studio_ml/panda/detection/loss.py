import torch
import torch.nn.functional as F

def generalized_cross_entropy(probs, targets, q=0.7):
    return (1 - torch.pow(probs[torch.arange(probs.size(0)), targets], q)) / q

def loss_coteaching(probs_1, probs_2, probs_3, targets, forget_rate, num_classes, q=0.7):

    loss_1 = generalized_cross_entropy(probs_1, targets, q)
    loss_2 = generalized_cross_entropy(probs_2, targets, q)
    loss_3 = generalized_cross_entropy(probs_3, targets, q)

    combined_loss_32 = 0.5 * loss_3 + 0.5 * loss_2
    combined_loss_13 = 0.5 * loss_1 + 0.5 * loss_3
    combined_loss_21 = 0.5 * loss_2 + 0.5 * loss_1

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(targets))

    ind_1_update = torch.argsort(combined_loss_32)[:num_remember]
    ind_2_update = torch.argsort(combined_loss_13)[:num_remember]
    ind_3_update = torch.argsort(combined_loss_21)[:num_remember]

    final_loss_1 = generalized_cross_entropy(probs_1[ind_1_update], targets[ind_1_update], q)
    final_loss_2 = generalized_cross_entropy(probs_2[ind_2_update], targets[ind_2_update], q)
    final_loss_3 = generalized_cross_entropy(probs_3[ind_3_update], targets[ind_3_update], q)

    return final_loss_1.mean(), final_loss_2.mean(), final_loss_3.mean()
