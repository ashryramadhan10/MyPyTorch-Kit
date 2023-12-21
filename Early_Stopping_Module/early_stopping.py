import numpy as np
import torch
from pathlib import Path

class EarlyStopping:
    def __init__(self,
                 max_patience: int,
                 delta: float,
                 checkpoints_dir: Path):
        self.max_patience = max_patience
        self.delta = delta
        self.checkpoints_dir = checkpoints_dir
        self.patience = 0
        self.minimum_loss = None
        
        self.isNeedToStop = False
        self.EARLY_STOPPING_BOUNDARY = 0.3
        self.MINIMUM_LOSS_TO_SAVE_WEIGHTS = 0.1
        
        # create checkpoints directory
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
    def __call__(self, loss, model, epoch):
        if self.minimum_loss is None:
            self.minimum_loss = loss
        
        elif loss > self.minimum_loss + self.delta:
            
            # if we have current minimum loss is below or equals to EARLY_STOPPING_BOUNDARY
            # then we could trigger the early stopping mechanism
            if self.minimum_loss <= self.EARLY_STOPPING_BOUNDARY:
                self.patience += 1
                print(f'EarlyStopping patience: {self.patience} out of {self.max_patience}')
                if self.patience >= self.max_patience:
                    self.isNeedToStop = True
        else:
            if loss <= self.MINIMUM_LOSS_TO_SAVE_WEIGHTS:
                model_filename = str(f"{model.model_name}_checkpoint_{epoch}.pt")
                self.save_model_weights(loss, model, model_filename)
                
            self.minimum_loss = loss
            self.patience = 0
            
    def save_model_weights(self, current_loss, model, filename):
        torch.save(model.state_dict(), self.path / filename)
        print(f'Validation loss decreased ({self.minimum_loss:.6f} --> {current_loss:.6f}). Saving model...')
                
