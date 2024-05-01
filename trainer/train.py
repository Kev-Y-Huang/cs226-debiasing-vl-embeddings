import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


class Trainer:
    def __init__(self, predictor, adversary, gender_vector):
        self.predictor = predictor
        self.adversary = adversary
        self.gender_vector = gender_vector
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

                # Calculate losses and backpropagate for predictor
                loss = criterion(outputs, embeds)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            train_loss /= len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{n_epochs}: Predictor Loss: {train_loss}")
        end = time.time()
        print(f"Pre-training completed in {end - start} seconds!")

    def pretrain_adversary(self, train_loader):
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

                # Calculate losses and backpropagate for predictor
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
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
            X, y, z, test_size=0.1, random_state=42, stratify=z
        )

        # Convert the data to PyTorch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        z_train_tensor = torch.tensor(z_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        z_test_tensor = torch.tensor(z_test, dtype=torch.float32)

        # Create a PyTorch DataLoader for training and testing data
        train_dataset = torch.utils.data.TensorDataset(
            x_train_tensor, y_train_tensor, z_train_tensor
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        test_dataset = torch.utils.data.TensorDataset(
            x_test_tensor, y_test_tensor, z_test_tensor
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
    ):
        """
        Training step for adversarial learning.

        Args:
            inputs: Input data as embeddings of the first three words in each analogy
            embeds: Output data as embeddings of the fourth word in each analogy
            attribs: Protected attribute data as the projection of the output embeddings
            debiased: Whether to use the debiased algorithm

        Returns:
            train_p_loss: Training loss of the predictor model
            train_a_loss: Training loss of the adversary model
        """
        train_p_loss = 0.0
        train_a_loss = 0.0

        self.p_optimizer.zero_grad()
        self.a_optimizer.zero_grad()

        # Compute loss for predictor on analogy completion task
        embed_hat = self.predictor(inputs)
        p_loss = self.loss_fn(embed_hat, embeds)
        train_p_loss = p_loss.item()

        # If not debiased, only train using predictor loss
        if not debiased:
            p_loss.backward()
            self.p_optimizer.step()
            return train_p_loss, 0

        p_loss.backward(retain_graph=True)

        # Cloning gradients of prediction loss w.r.t. predictor parameters
        dW_LP = [torch.clone(p.grad.detach()) for p in self.predictor.parameters()]

        self.p_optimizer.zero_grad()
        self.a_optimizer.zero_grad()

        # Compute loss for adversary on protected attribute prediction task
        attrib_hat = self.adversary(embed_hat)
        a_loss = self.loss_fn(attrib_hat, attribs)
        a_loss.backward()

        # Cloning gradients of adversarial loss w.r.t. predictor parameters
        dW_LA = [torch.clone(p.grad.detach()) for p in self.predictor.parameters()]

        # Compute loss for adversary on protected attribute prediction task (NEW CHANGE?)
        self.a_optimizer.zero_grad()
        attrib_hat = self.adversary(embeds)
        a_loss = self.loss_fn(attrib_hat, attribs)
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

    def train(self, X, y, z, debiased=True):
        """
        Trains the predictor and adversary models.

        Args:
            X: Input data as embeddings of the first three words in each analogy
            y: Output data as embeddings of the fourth word in each analogy
            z: Protected attribute data as the projection of the output embeddings
            debiased: Whether to use the debiased algorithm (default: True)
        """
        # Split data into train and test
        train_loader, test_loader = self.prepare_data(X, y, z)

        # Set the number of training epochs
        n_epochs = 750

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
                    inputs, embeds, attribs, debiased
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
                    f"Train P Loss: {train_p_loss:.4f}, Train A Loss: {train_a_loss:.4f}"
                )
                print(f"Test P Loss: {test_p_loss:.4f}, Test A Loss: {test_a_loss:.4f}")

        end = time.time()
        print(f"Training completed in {end - start} seconds!")

        # Compute metrics about the learned predictor
        w = self.predictor.w.detach().clone().numpy()
        proj = np.dot(w.T, self.gender_vector)
        size = np.linalg.norm(w)
        print(f"Learned w has |w|={size} and <w,g>={proj}.")
