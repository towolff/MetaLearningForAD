import torch
from Experiments.FilteredDataset import FilteredDataset
from torch.nn.functional import mse_loss


class FineTuner:

    def __init__(self, fine_tune_data, k, model, optimizer, fine_tune_classes, num_iterations):
        self.fine_tune_data = fine_tune_data
        self.model = model
        self.optimizer = optimizer
        self.k = k
        self.fine_tune_classes = fine_tune_classes
        self.num_iterations = num_iterations
        self.all_losses_fine_tune = []
        self.avg_losses_fine_tune = []

    def _sample_fine_tune_data(self):
        """
        Sample k samples from n classes and set in class variable.
        :return: Nothing.
        """
        if not isinstance(self.fine_tune_data, torch.utils.data.TensorDataset):
            raise Exception('Fine Tune Data is not a TensorDataset!')

        # select classes
        if self.fine_tune_classes is not None:
            dataset = FilteredDataset(self.fine_tune_data, self.fine_tune_classes)
        else:
            dataset = self.fine_tune_data

        # select k samples
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=self.k)
        fine_tune_DataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, sampler=sampler)

        # draw distinct data
        self.X = []
        self.drift_label = []

        # sample unique
        for j, (x, y) in enumerate(fine_tune_DataLoader):
            self.X.append(x)
            self.drift_label.append(y)

    def fine_tune(self):
        self._sample_fine_tune_data()
        self.model.train()

        for _ in range(self.num_iterations):
            inner_loss = 0.0
            avg_inner_loss = 0.0
            for i, (x, y) in enumerate(zip(self.X, self.drift_label)):
                x = torch.autograd.Variable(x)
                self.optimizer.zero_grad()

                pred = self.model(x)
                loss = mse_loss(pred, x)
                self.all_losses_fine_tune.append(loss.item())

                inner_loss += loss.item()
                avg_inner_loss = inner_loss / self.k

                # Backpropagation
                loss.backward()
                self.optimizer.step()

            self.avg_losses_fine_tune.append(avg_inner_loss)