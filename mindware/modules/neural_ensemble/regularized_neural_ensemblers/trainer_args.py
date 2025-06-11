import torch

class TrainerArgs:
    """
    Class to hold the arguments for the trainer.
    """
    def __init__(self, batch_size : int = 32, 
                 epochs: int =10, 
                 lr : float =0.001, 
                 checkpoint_name : str ="neural_ensembler.pt",
                 clip_value : float = 1.0,
                 device : torch.device = None,
                 task_id: str = None,
                 task_type: int = None):
        """
        Initialize the TrainerArgs class.

        Args:
            model: The model to be trained.
            task_type: The type of task (e.g., 'classification', 'regression').
            batch_size: The batch size for training.
            lr: The learning rate for the optimizer.
            epochs: The number of epochs for training.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.task_type = task_type
        self.lr = lr
        self.checkpoint_name = f't{task_type}_{task_id}_{checkpoint_name}'
        self.clip_value = clip_value
        self.device = device if device is not None else \
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device