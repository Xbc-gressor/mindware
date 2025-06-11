import torch
import numpy as np

from .model import NeuralEnsembler
from .trainer_args import TrainerArgs
from mindware.components.utils.constants import CLS_TASKS, REGRESSION

class Trainer:
    """
    Class to train the NeuralEnsembler model.
    """

    def __init__(self, model: NeuralEnsembler, 
                 trainer_args: TrainerArgs = None
    ):
        """
        Initialize the Trainer class.
        Args:
            model (NeuralEnsembler): The model to be trained.
            trainer_args (TrainerArgs): The arguments for the trainer.
        """

        self.model = model
        self.trainer_args = trainer_args if trainer_args is not None else TrainerArgs()
        self.__dict__.update(self.trainer_args.__dict__)

    def fit(self, X: np.array, y: np.array) -> None:
        """Train the model on the given data.
        Args:
            X (np.ndarray): Input data of shape (num_samples, num_base_functions, num_classes).
            y (np.ndarray): Target data of shape (num_samples, num_classes). 
        """

        loader = self._get_loader(X, y)	
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_lst = []
        self.model.train()
        for epoch in range(self.epochs):
            _loss_sum = 0
            _num = 0
            self.model.train()
            for base_functions, target in loader:
                _sub_num = len(base_functions)
                optimizer.zero_grad()
                output, _ = self.model(base_functions)
                loss = self._loss_fn(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                optimizer.step()
                _loss_sum += loss.item() * _sub_num
                _num += _sub_num
            loss_lst.append(_loss_sum / _num)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {_loss_sum / _num}") 

        if self.model.task_type == REGRESSION:
            self.model.set_y_scale(self.y_scale)

        self.model.save_checkpoint(self.checkpoint_name)
        return loss_lst
    
    def _get_loader(self, X, y):
        """
        Create a DataLoader for the given data.
        Args:
            X (np.ndarray): Input data of shape (num_samples, num_base_functions, num_classes).
            y (np.ndarray): Target data of shape (num_samples, num_classes).
        Returns:
            loader (torch.utils.data.DataLoader): DataLoader for the given data.
        """

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        if self.model.task_type == REGRESSION:
            self.y_scale = torch.abs(X).mean()
            y /= self.y_scale.item()
            X /= self.y_scale.item()

        elif self.model.task_type in CLS_TASKS:
            y = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, y)	
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return loader

    def _loss_fn(self, output: torch.tensor, target: torch.tensor) -> torch.tensor:
        """
        Compute the loss between the model output and the target.
        Args:   
            output (torch.Tensor): Model output of shape (num_samples, num_classes) or (num_samples).
            target (torch.Tensor): Target Tensor of shape (num_samples).
        Returns:
            loss (torch.Tensor): Computed loss.
        """

        if self.model.task_type == REGRESSION:
            loss = torch.nn.MSELoss()(output.reshape(-1), target.reshape(-1))
        elif self.model.task_type in CLS_TASKS:
            logits = self._get_logits_from_probabilities(output)
            loss = torch.nn.CrossEntropyLoss()(logits, target.reshape(-1))
        else:
            raise ValueError(f"Unknown task type: {self.model.task_type}")
        return loss

    def _get_logits_from_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Get the logits given the probabilities.
        Args:
            probabilities (torch.Tensor): probability Tensor of shape (num_ensembles, num_pipelines, num_samples, num_classes).
        Returns:
            logits (torch.Tensor): probability Tensor of shape (num_ensembles, num_pipelines, num_samples, num_classes)

        """

        log_p = torch.log(probabilities + 10e-8)
        C = -log_p.mean(-1)
        logits = log_p + C.unsqueeze(-1)
        return logits
    
