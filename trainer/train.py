import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from utils.preprocess import project_onto_subspace


class Trainer:
    def __init__(self, predictor, adversary, gender_vector, word_embeds, w2i):
        self.predictor = predictor
        self.adversary = adversary
        self.gender_vector = gender_vector
        self.word_embeds = word_embeds
        self.w2i = w2i
        self.p_optimizer = optim.Adam(self.predictor.parameters(), lr=0.0001)
        self.a_optimizer = optim.Adam(self.adversary.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The seed value to set.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def pretrain_predictor(self, train_loader):
        """
        Pre-trains the predictor model.

        Args:
            train_loader: DataLoader for training data
        """
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.predictor.parameters(), lr=2 ** (-16))

        # Set the number of training epochs
        n_epochs = 3

        # Training loop
        start = time.time()
        self.predictor.train()
        for epoch in range(n_epochs):
            train_loss = 0.0
            for inputs, embeds, _ in train_loader:
                optimizer.zero_grad()
                outputs = self.predictor(inputs)
                loss = criterion(outputs, embeds)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            train_loss /= len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{n_epochs}: Predictor Loss: {train_loss}")
        end = time.time()
        print(f"Pre-training completed in {end - start} seconds!")

    def pretrain_adversary(self, train_loader):
        """
        Pre-trains the adversary model.

        Args:
            train_loader: DataLoader for training data
        """
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.adversary.parameters(), lr=2 ** (-16))

        # Set the number of training epochs
        n_epochs = 3

        # Training loop
        start = time.time()
        self.adversary.train()
        for epoch in range(n_epochs):
            train_loss = 0.0
            for _, embeds, attribs in train_loader:
                optimizer.zero_grad()
                outputs = self.adversary(embeds)
                loss = criterion(outputs, attribs)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            train_loss /= len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{n_epochs}: Adversary Loss: {train_loss}")
        end = time.time()
        print(f"Pre-training completed in {end - start} seconds!")

    def prepare_data(self, X, y, z):
        """
        Prepares the data for training and testing using PyTorch DataLoader.

        Args:
            X: Input data as embeddings of the first three words in each analogy
            y: Output data as embeddings of the fourth word in each analogy
            z: Protected attribute data as the projection of the output embeddings

        Returns:
            train_loader: DataLoader for training data
            test_loader: DataLoader for testing data
        """
        # Split data into train and test
        dataset = torch.utils.data.TensorDataset(X, y, z)
        train_len = int(0.9 * len(dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_len, len(dataset) - train_len]
        )

        # Create a PyTorch DataLoader for training and testing data
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False
        )

        return train_loader, test_loader

    def adversarial_train_step(
        self,
        inputs,
        embeds,
        attribs,
        debiased,
        recompute_subspace,
    ):
        """
        Training step for adversarial learning.

        Args:
            inputs (torch.tensor): Input data as embeddings of the first three words in each analogy
            embeds (torch.tensor): Output data as embeddings of the fourth word in each analogy
            attribs (torch.tensor): Protected attribute data as the projection of the output embeddings
            debiased (bool): Whether to use the debiased algorithm
            recompute_subspace (bool): Whether to recompute the subspace

        Returns:
            train_p_loss: Training loss of the predictor model
            train_a_loss: Training loss of the adversary model
        """
        train_p_loss = 0.0
        train_a_loss = 0.0

        self.p_optimizer.zero_grad()
        self.a_optimizer.zero_grad()

        # Generate y_hat and compute L_P(y_hat, y)
        pred = self.predictor(inputs)
        p_loss = self.loss_fn(pred, embeds)
        train_p_loss = p_loss.item()

        # If not debiased, update predictor only using dW_LP
        if not debiased:
            p_loss.backward()
            self.p_optimizer.step()
            return train_p_loss, 0

        # Retain graph so that can compute dW_LA
        p_loss.backward(retain_graph=True)

        # Cloning gradients of prediction loss w.r.t. W
        dW_LP = [p.grad.detach().clone() for p in self.predictor.parameters()]

        self.p_optimizer.zero_grad()
        self.a_optimizer.zero_grad()

        # Generate z_hat and compute L_A(z_hat, z)
        attrib_pred = self.adversary(pred)
        a_loss = self.loss_fn(attrib_pred, attribs)
        train_a_loss = a_loss.item()
        a_loss.backward()

        # Cloning gradients of adversary loss w.r.t. W
        dW_LA = [p.grad.detach().clone() for p in self.predictor.parameters()]

        if recompute_subspace:
            # Update adversary using dW_LA for z_hat generate from true y
            self.a_optimizer.zero_grad()
            attrib_pred = self.adversary(embeds)
            a_loss = self.loss_fn(attrib_pred, attribs)
            train_a_loss = a_loss.item()
            a_loss.backward()

        for i, p in enumerate(self.predictor.parameters()):
            # Normalize dW_LA
            unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + torch.finfo(float).tiny)
            # Project dW_LP onto dW_LA
            proj = torch.sum(torch.inner(unit_dW_LA, dW_LP[i]))
            # Calculate predictor gradients according to Zhang et al. (2018)
            p.grad = dW_LP[i] - (proj * unit_dW_LA) - (dW_LA[i])

        self.p_optimizer.step()
        self.a_optimizer.step()

        return train_p_loss, train_a_loss

    def train(self, X, y, z, debiased=True, n_epochs=250, recompute_subspace=False):
        """
        Trains the predictor and adversary models.

        Args:
            X: Input data as embeddings of the first three words in each analogy
            y: Output data as embeddings of the fourth word in each analogy
            z: Protected attribute data as the projection of the output embeddings
            debiased: Whether to use the debiased algorithm (default: True)
            n_epochs: Number of training epochs (default: 250)
        """
        # Split data into train and test
        train_loader, test_loader = self.prepare_data(X, y, z)

        # Training loop
        start = time.time()
        for epoch in range(n_epochs):
            self.predictor.train()
            self.adversary.train()
            train_p_loss = 0.0
            train_a_loss = 0.0

            # Train the models
            for inputs, embeds, attribs in train_loader:
                p_loss, a_loss = self.adversarial_train_step(
                    inputs, embeds, attribs, debiased, recompute_subspace
                )
                train_p_loss += p_loss
                train_a_loss += a_loss

            train_p_loss /= len(train_loader.dataset)
            train_a_loss /= len(train_loader.dataset)

            self.predictor.eval()
            self.adversary.eval()
            test_p_loss = 0.0
            test_a_loss = 0.0
            with torch.no_grad():
                # Test the models
                for inputs, embeds, attribs in test_loader:
                    outputs = self.predictor(inputs)
                    p_loss = self.loss_fn(outputs, embeds)
                    test_p_loss += p_loss.item()

                    if debiased:
                        labels = self.adversary(outputs)
                        a_loss = self.loss_fn(labels, attribs)
                        test_a_loss += a_loss.item()
            test_p_loss /= len(test_loader.dataset)
            test_a_loss /= len(test_loader.dataset)

            # Log the loss values
            if epoch % 50 == 49:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(
                    f"Train P Loss: {train_p_loss:.4E}, Train A Loss: {train_a_loss:.4E}"
                )
                print(f"Test P Loss: {test_p_loss:.4E}, Test A Loss: {test_a_loss:.4E}")
            if recompute_subspace:
                # Recompute the embeddings every epoch
                self.word_embeds = self.predictor.predict_batch(self.word_embeds)

                # Recompute training data embeddings using predictor modl
                old_shape = X.size()
                X_embeds = torch.flatten(X, end_dim=1)
                new_X_embeds = self.predictor.predict_batch(X_embeds)
                X = torch.reshape(new_X_embeds, old_shape)
                y = self.predictor.predict_batch(y)

                # Recompute the subspace every 10 epochs
                if epoch % 10 == 0:
                    z, self.gender_vector = project_onto_subspace(
                        self.word_embeds, self.w2i, y
                    )

                train_loader, test_loader = self.prepare_data(X, y, z)

        end = time.time()
        print(f"Training completed in {end - start} seconds!")

        # Recomput word_embeddings after training
        self.word_embeds = self.predictor.predict_batch(self.word_embeds)

        if hasattr(self.predictor, 'w'):
            # Compute metrics about the learned predictor if simple predictor
            w = torch.flatten(self.predictor.w)
            proj = torch.dot(w, self.gender_vector)
            size = torch.norm(w)
            print(f"Learned w has |w|={size:.4f} and <w,g>={proj}.")
