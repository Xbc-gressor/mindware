import os
import copy
from pathlib import Path

import torch 
import torch.nn as nn
import numpy as np
from mindware.components.utils.constants import REGRESSION

class NeuralEnsembler(nn.Module):
    """
    NeuralEnsembler class for model averaging and stacking.
    This class implements a neural network that can be used for model averaging
    and stacking of base functions.
    """ 
    def __init__(
        self,
        num_base_functions,
        hidden_dim=32,
        output_dim=1,
        num_layers=3,
        dropout_rate=0,
        num_heads=1,
        inner_batch_size=10,
        omit_output_mask=False,
        uniform_mask=True,
        task_type="classification",
        mode="model_averaging",
        num_classes=None,
        device=None,
        **kwargs
    ):
        """
        Initialize the NeuralEnsembler class.
        Args:
            num_base_functions (int): Number of base functions.
            hidden_dim (int): Hidden dimension for the neural network.
            output_dim (int): Output dimension for the neural network.
            num_layers (int): Number of layers in the neural network.
            dropout_rate (float): Dropout rate for the neural network.
            num_heads (int): Number of heads for the attention mechanism.
            inner_batch_size (int): Inner batch size for processing classes.
            omit_output_mask (bool): Whether to omit the output mask.
            uniform_mask (bool): Whether to use a uniform mask across samples for dropout.
            task_type (str): Type of task ('classification' or 'regression').
            mode (str): Mode of operation ('model_averaging' or 'stacking').
            num_classes (int): Number of classes for classification tasks.
        """

        super().__init__()

        self.num_base_functions = num_base_functions
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.inner_batch_size = inner_batch_size
        self.omit_output_mask = omit_output_mask
        self.mode = mode
        self.task_type = task_type
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.uniform_mask = uniform_mask
        self.y_scale = 1
        self.device = device if device is not None else \
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert self.num_layers > 0

        if self.mode == "stacking":
            self.output_dim = 1
        else:
            self.output_dim = num_base_functions

        self.build_modules()

    def build_modules(self):
        """
        Build the modules for the neural network.
        """
        layers_dim = [self.num_base_functions] + [self.hidden_dim] * (self.num_layers-1)
        first_module = []
        for i in range(len(layers_dim) - 1):
            first_module.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
            first_module.append(nn.ReLU())
        self.first_module = nn.Sequential(*first_module)
        self.out_layer = nn.Linear(self.hidden_dim, self.output_dim)

        if self.mode == "model_averaging":
            self.second_module = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU())


    def _get_mask_and_scaling_factor(self, x: torch.Tensor):
        """
        Get the mask and scaling factor for the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (num_samples, num_classes, num_base_functions).
        Returns:
            mask (torch.Tensor): Mask tensor of shape (num_samples, num_classes, num_base_functions).
            scaling_factor (float): Scaling factor for the mask.
        """
        scaling_factor = 1
        mask = None
        num_samples, num_classes, num_base_functions = x.shape
        if self.training and self.dropout_rate > 0:
            scaling_factor = 1./(1.- self.dropout_rate)

            if self.uniform_mask:
                mask= (torch.rand(size=(num_base_functions,)) > self.dropout_rate).float().to(x.device)
                for i, dim in enumerate([num_samples, num_classes]):
                    mask = torch.repeat_interleave(
                        mask.unsqueeze(i), dim, dim=i
                    )  
            else:
                mask= (torch.rand(size=(num_samples, 1, num_base_functions,)) > self.dropout_rate).float().to(x.device)
                mask = torch.repeat_interleave(
                    mask, num_classes, dim=1
                )

        return mask, scaling_factor

    def _batched_forward_across_classes(self, x: torch.Tensor, base_functions: torch.Tensor):
        """
        Perform a batched forward pass across classes.
        Args:
            x (torch.Tensor): Input tensor of shape (num_samples, num_classes, num_base_functions).
            base_functions (torch.Tensor): Base functions tensor of shape (num_samples, num_classes, num_base_functions).
        Returns:
            w (torch.Tensor): Weights tensor of shape (num_samples, num_classes, num_base_functions).
            mask (torch.Tensor): Mask tensor of shape (num_samples, num_classes, num_base_functions).
        """

        num_samples, num_classes, num_base_functions = x.shape
        mask, scaling_factor = self._get_mask_and_scaling_factor(x)

        w = []
        idx = np.arange(num_classes)
        for i in range(0, num_classes, self.inner_batch_size):
            range_idx = idx[range(i, min(i + self.inner_batch_size, num_classes))]
            if mask is not None:
                temp_x = (x[:,range_idx]*mask[:,range_idx])*scaling_factor
                base_functions[:,range_idx] = base_functions[:,range_idx]*mask[:,range_idx]
            else:
                temp_x = x[:,range_idx]
            temp_w = self.first_module(temp_x)
            w.append(temp_w)
        w = torch.cat(w, axis=1)
        return w, mask
    
    def forward(
        self, x: torch.Tensor
    ):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (num_samples, num_classes, num_base_functions).
        Returns:
            x (torch.Tensor): Output tensor of shape (num_samples, num_classes).
            w_norm (torch.Tensor): Normalized weights tensor of shape (num_samples, num_classes, num_base_functions).
        """


        num_samples, num_classes, num_base_functions = x.shape
        base_functions = copy.deepcopy(x)
        w, mask = self._batched_forward_across_classes(x, base_functions)

        if self.mode == "stacking":
            w = self.out_layer(w)
            x = w.squeeze(-1)
            if self.task_type == "classification":
                x = torch.nn.functional.softmax(x, dim=-1)
            w_norm = None

        elif self.mode == "model_averaging":
            w = w.mean(axis=1)
            w = self.second_module(w)
            w = torch.repeat_interleave(
                w.unsqueeze(1), num_classes, dim=1
            )
            w = self.out_layer(w)

            if (mask is not None) and (not self.omit_output_mask):
                w = w.masked_fill(mask == 0, -1e9)

            w_norm = torch.nn.functional.softmax(w, dim=-1)
            x = torch.multiply(base_functions, w_norm).sum(axis=-1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # x.shape: [num_samples, num_classes]
        # w_norm.shape : [num_samples, num_classes, num_base_functions]
        return x, w_norm
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (num_samples, num_classes, num_base_functions).
        Returns:
            x (torch.Tensor): Output tensor of shape (num_samples, num_classes).
        """
        self.eval()
        with torch.no_grad():
            device = self.first_module[0].weight.device
            x = torch.tensor(x, dtype=torch.float32).to(device)

            y_pred = self.forward(x/self.y_scale)[0]

        if self.task_type == REGRESSION:
            y_pred = y_pred * self.y_scale

        y_pred = y_pred.cpu().numpy()
        return y_pred
    

    def _get_config(self):
        """
        Get the configuration of the model.
        Returns:
            config (dict): Configuration dictionary.
        """

        config = {
            "num_base_functions": self.num_base_functions,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "num_heads": self.num_heads,
            "inner_batch_size": self.inner_batch_size,
            "omit_output_mask": self.omit_output_mask,
            "task_type": self.task_type,
            "mode": self.mode,
            "y_scale": self.y_scale
        }
        return config
    
    def set_y_scale(self, y_scale: float):
        """
        Set the y_scale attribute of the model.
        Args:
            y_scale (float): The y_scale value to set.
        """
        self.y_scale = y_scale

    def save_checkpoint(self, checkpoint_name: str = "neural_ensembler.pt"):
        """ 
        Save the model checkpoint to the specified file.
        Args:
            checkpoint_name (str): The name of the checkpoint file.
        """ 
        complete_path = Path(os.path.abspath(__file__)).parent.parent / "checkpoints"
        complete_path.mkdir(parents=True, exist_ok=True)
        config = self._get_config()
        torch.save({
                    'config': config,
                    'model_state_dict': self.state_dict(),
                    }, complete_path / checkpoint_name)

    @staticmethod
    def load_checkpoint(cls, checkpoint_name: str = "neural_ensembler.pt"):

        """
        Load a checkpoint from the specified file.
        Args:
            cls: The class of the model to load.
            checkpoint_name (str): The name of the checkpoint file.
        Returns:
            model: The loaded model.
        """

        try:
            complete_path = Path(os.path.abspath(__file__)).parent.parent / "checkpoints"
            checkpoint = torch.load(complete_path / checkpoint_name)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found in {complete_path}")
        
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model