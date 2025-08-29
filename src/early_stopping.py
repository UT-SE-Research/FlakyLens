import numpy as np
import torch

#This one is checking of validation_loss
#class EarlyStopping:
#    def __init__(self, patience, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
#        """
#        Args:
#            patience (int): How long to wait after last time validation loss improved.
#                            Default: 5
#            verbose (bool): If True, prints a message for each validation loss improvement. 
#                            Default: False
#            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                            Default: 0
#            path (str): Path for the checkpoint to be saved to.
#                            Default: 'checkpoint.pt'
#            trace_func (function): trace print function.
#                            Default: print
#        """
#        self.patience = patience
#        self.verbose = verbose
#        self.counter = 0
#        self.best_score = None
#        self.early_stop = False
#        self.val_loss_min = np.Inf
#        self.delta = delta
#        self.path = path
#        self.trace_func = trace_func
#
#    def __call__(self, val_loss, model):
#
#        score = -val_loss
#
#        if self.best_score is None:
#            self.best_score = score
#            self.save_checkpoint(val_loss, model)
#        elif score < self.best_score + self.delta:
#            self.counter += 1
#            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#            if self.counter >= self.patience:
#                self.early_stop = True
#        else:
#            self.best_score = score
#            self.save_checkpoint(val_loss, model)
#            self.counter = 0
#
#    def save_checkpoint(self, val_loss, model):
#        '''Saves model when validation loss decrease.'''
#        if self.verbose:
#            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
#        #torch.save(model.state_dict(), self.path)
#        self.val_loss_min = val_loss

import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """ 
        Args:
            patience (int): How many epochs to wait after last improvement before stopping.
            verbose (bool): If True, prints a message for each improvement.
            delta (float): Minimum change in monitored metric to count as improvement.
            path (str): Path for saving the best model.
            trace_func (function): Function used for logging. Default: print.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0 
        self.best_score = None
        self.early_stop = False
        self.best_f1 = 0.0  # ✅ Track best F1-score instead of loss
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, valid_f1, model):
        """
        Monitor validation F1-score and trigger early stopping.
        
        Args:
            valid_f1 (float): Current epoch validation F1-score.
            model (torch.nn.Module): Model to save if it improves.
        """
        score = valid_f1  # ✅ Track F1-score instead of validation loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(valid_f1, model)
        elif score <= self.best_score + self.delta:  # No improvement
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # Improvement found
            self.best_score = score
            self.save_checkpoint(valid_f1, model)
            self.counter = 0  # Reset patience counter

    def save_checkpoint(self, valid_f1, model):
        '''Saves model when validation F1-score improves.'''
        if self.verbose:
            self.trace_func(f'Validation F1-score improved ({self.best_f1:.6f} --> {valid_f1:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)  # ✅ Save the model
        self.best_f1 = valid_f1  # ✅ Update best F1-score

