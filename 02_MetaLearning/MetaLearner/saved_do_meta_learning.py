def do_meta_learning(self):
    tqdm_bar = tqdm(range(self.num_iterations))

    for _ in tqdm_bar:

        iteration_error = 0.0
        for _ in range(self.tasks_per_step):
            learner = self.meta_model.clone()
            train_task = self.train_task_data.sample()
            evaluation_task = self.eval_task_data.sample()

            data_adapt, labels_adapt = train_task
            data_eval, labels_eval = evaluation_task

            data_adapt = data_adapt.to(self.device)
            data_eval = data_eval.to(self.device)

            # Fast Adaptation
            for step in range(self.adaption_steps):
                train_error = self.mse_loss(learner(data_adapt), data_adapt)
                self.adaption_loss.append(train_error)
                learner.adapt(train_error)

            # Compute validation loss
            predictions = learner(data_eval)
            valid_error = self.mse_loss(predictions, data_eval)
            valid_error /= len(data_eval)
            iteration_error += valid_error
            self.evaluation_loss.append(valid_error)

        # Update status bar
        tqdm_bar.set_description("Iteration Error: {:.4f}, Validation Error: {}".format(iteration_error.item(),
                                                                                        valid_error.item()))

        # Take the meta-learning step
        self.optimizer.zero_grad()
        iteration_error.backward()
        self.optimizer.step()

        # Save Weights for Analysis
        if self.save_weights:
            self._update_weight_matrices_of_meta_model()
