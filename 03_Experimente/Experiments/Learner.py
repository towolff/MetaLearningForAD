import learn2learn as l2l
import torch

from tqdm import tqdm


class Learner:
    """
    Learner Class. Wraps the Meta-Learning Phase in learn2learn to easily evaluate different hyperparameters.
    """

    def __init__(self, model, train_task_dataset=None, eval_task_dataset=None,
                 adaption_steps=10, use_cuda=False, meta_lr=1e-3, lr=4e-3, optimizer='Adam',
                 first_order=False, num_iterations=1000, save_weights=False, done_training_phase=False):
        """
        Initilize Learner Class with values.
        :param model: The PyTorch Model to train with Meta-Learning.
        :param train_task_dataset: l2l MetaDataset for Training.
        :param eval_task_dataset: l2l MetaDataset for evaluation and adaption.
        :param adaption_steps: Number of Samples to use for adaption.
        :param use_cuda: True if use cuda.
        :param meta_lr: Learning Rate of the Meta-Model
        :param lr: Learning Rate of the Model.
        :param optimizer: String (Adam or SGD) which PyTorch Optimizer to use.
        :param first_order: True if use First Order in MAML.
        :param num_iterations: Number of Iteration for the Outer-Learning Loop.
        :param save_weights: True if save weights.
        :param done_training_phase: Set to True, when Meta-Learning Phase is done.
        """
        self.train_task_data = train_task_dataset
        self.eval_task_data = eval_task_dataset
        self.model = model
        self.meta_model = None
        self._init_meta_model(meta_lr, first_order)
        self.optimizer = None
        self._init_optimizer(lr=lr, optimizer=optimizer)
        self.num_iterations = num_iterations
        self.adaption_steps = adaption_steps
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.adaption_loss = []
        self.evaluation_loss = []
        self.iteration_error_list = []
        self.mse_loss = torch.nn.functional.mse_loss
        self.weight_matrices = {'encoder': [], 'decoder': []}
        self.weight_matrices_meta_model = {'encoder': [], 'decoder': []}
        self.save_weights = save_weights
        self.done_training_phase = done_training_phase

    def _update_weight_matrices(self):
        self.weight_matrices['encoder'].append(self.model.encoder.state_dict())
        self.weight_matrices['decoder'].append(self.model.decoder.state_dict())

    def _update_weight_matrices_of_meta_model(self):
        self.weight_matrices_meta_model['encoder'].append(self.meta_model.encoder.state_dict())
        self.weight_matrices_meta_model['decoder'].append(self.meta_model.decoder.state_dict())

    def _init_meta_model(self, meta_lr, first_order):
        self.meta_model = l2l.algorithms.MAML(self.model, lr=meta_lr, first_order=first_order)

    def _init_optimizer(self, lr, optimizer):
        if optimizer == 'Adam' or optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=lr)
        elif optimizer == 'SGD' or optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.meta_model.parameters(), lr=lr)
        else:
            raise NotImplementedError

    def start_learning_phase(self):
        """
        Train initial model with random weights!
        :return: nothing!
        """

        tqdm_bar = tqdm(range(self.num_iterations))
        self.meta_model.module.train()

        for _ in tqdm_bar:
            iteration_error = 0.0

            # For Loop samples a number of tasks from the train data!
            for task in self.train_task_data:
                learner = self.meta_model.clone()

                # Split data for adaption and evaluation
                adaption_data = task[0]
                adaption_data = adaption_data.to(self.device)

                eval_data = self.eval_task_data.sample()[0]
                eval_data = eval_data.to(self.device)

                # Fast Adaption
                for step in range(self.adaption_steps):
                    train_error = self.mse_loss(learner(adaption_data), adaption_data)
                    self.adaption_loss.append(train_error.item())
                    learner.adapt(train_error)

                # Compute validation loss
                predictions = learner(eval_data)
                valid_error = self.mse_loss(predictions, eval_data)
                valid_error /= len(eval_data)
                iteration_error += valid_error
                self.evaluation_loss.append(valid_error.item())

            # Update status bar
            tqdm_bar.set_description("Adaption Error: {:.6f}, Validation Error: {:.6f}".format(train_error.item(),
                                                                                            valid_error.item()))

            # Take the meta-learning step
            self.iteration_error_list.append(iteration_error.item())
            self.optimizer.zero_grad()
            iteration_error.backward()
            self.optimizer.step()

            # Save Weights for Analysis
            if self.save_weights:
                self._update_weight_matrices_of_meta_model()

        self.done_training_phase = True


