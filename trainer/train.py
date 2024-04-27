import time

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model.simple_gan import *


class Trainer:
    def __init__(self, wv, w2i):
        self.wv = wv
        self.w2i = w2i
        self.gender_vector = None

    def project_onto_subspace(self, y):
        # Calculate "gender direction"
        # Female-Male word pairs
        pairs = [
            ("woman", "man"),
            ("her", "his"),
            ("she", "he"),
            ("aunt", "uncle"),
            ("niece", "nephew"),
            ("daughters", "sons"),
            ("mother", "father"),
            ("daughter", "son"),
            ("granddaughter", "grandson"),
            ("girl", "boy"),
            ("stepdaughter", "stepson"),
            ("mom", "dad"),
        ]
        # Calculate the difference between female and male word embeddings
        diff = np.array([self.wv[self.w2i[wf]] - self.wv[self.w2i[wm]] for wf, wm in pairs])
        
        # Bias subspace is defined by top principal component of the differences
        cov = np.cov(np.array(diff).T)
        evals, evecs = np.linalg.eig(cov)
        dir = np.real(evecs[:, np.argmax(evals)])
        self.gender_vector = dir / np.linalg.norm(dir)

        # Get projection of output embeddings onto gender subspace
        z = np.array([np.dot(y_, self.gender_vector) for y_ in y])
        return z

    def preprocess(self):
        """
        Preprocesses the analogy data to produce input, output, and protected attribute data.

        Returns:
            X: Input data as embeddings of the first three words in each analogy
            y: Output data as embeddings of the fourth word in each analogy
            a: Protected attribute data as the cosine similarity of each output embedding
        """
        # Load analogy data
        url = 'http://download.tensorflow.org/data/questions-words.txt'
        # Family category includes gender dynamics
        category = 'family'
        r = requests.get(url, allow_redirects=False)
        lines = r.text.split('\n')
        gender_pairs = set()
        valid_category = False
        for line in lines:
            sp = line.split(' ')
            # Start of category will be preceded by the line ": family"
            if not valid_category:
                valid_category = sp[1] == category
            else:
                # Each analogy will be formatted as "a b c d" where a:b::c:d
                if len(sp) == 4:
                    gender_pairs.add((sp[0], sp[1]))
                    gender_pairs.add((sp[2], sp[3]))
                # End of category will be a new category declaration ": category"
                else:
                    break

        path = "/n/home12/kevhuang/projects/cs226-debiasing-vl-embeddings/Adversarial-Gender-debiased-WE/data/equalize_pairs.json"
        with open(path, 'r') as f:
            import json
            equalize_pairs = json.load(f)
            for pair in equalize_pairs:
                gender_pairs.add((pair[0], pair[1]))

        gender_pairs = list(gender_pairs)
        print(f"{len(gender_pairs)} total pairs")

        analogies = []
        for i, pair1 in enumerate(gender_pairs):
            for pair2 in gender_pairs[:i] + gender_pairs[i + 1:]:
                analogies.append(pair1 + pair2)
        print(f"{len(analogies)} analogies!")

        X = []
        y = []
        for analogy in tqdm(analogies):
            x = []
            for word in analogy[:-1]:
                x.append(self.wv[self.w2i[word]])
            X.append(x)
            y.append(self.wv[self.w2i[analogy[-1]]])
        X = np.array(X)
        y = np.array(y)
        z = self.project_onto_subspace(y)

        return X, y, z

    def pretrain_predictor(self, predictor, train_loader):
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(predictor.parameters(), lr=2**(-16))

        # Set the number of training epochs
        n_epochs = 3

        # Training loop
        start = time.time()
        predictor.train()
        for _ in range(n_epochs):
            for inputs, embeds, _ in train_loader:
                optimizer.zero_grad()

                outputs = predictor(inputs)

                # Calculate losses and backpropagate for predictor
                loss = criterion(outputs, embeds)
                loss.backward()
                optimizer.step()
        end = time.time()
        print(f"Pre-training completed in {end - start} seconds!")

    def pretrain_adversary(self, adversary, train_loader):
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(adversary.parameters(), lr=2**(-16))

        # Set the number of training epochs
        n_epochs = 3

        # Training loop
        start = time.time()
        adversary.train()
        for _ in range(n_epochs):
            for _, embeds, attribs in train_loader:
                optimizer.zero_grad()

                outputs = adversary(embeds)

                # Calculate losses and backpropagate for predictor
                loss = criterion(outputs, attribs)
                loss.backward()
                optimizer.step()
        end = time.time()
        print(f"Pre-training completed in {end - start} seconds!")
    
    def prepare_data(self, X, y, z):
        # Split data into train and test
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X,
                                                            y,
                                                            z,
                                                            test_size=0.1,
                                                            random_state=42,
                                                            stratify=z)

        # Convert the data to PyTorch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        z_train_tensor = torch.tensor(z_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        z_test_tensor = torch.tensor(z_test, dtype=torch.float32)

        # Create a PyTorch DataLoader for training and testing data
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor, z_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor, z_test_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader

    def train(self, X, y, a, predictor, adversary, debiased=True):
        # Split data into train and test
        train_loader, test_loader = self.prepare_data(X, y, a)

        # Define the loss function and optimizer
        mse_loss = nn.MSELoss()

        p_optimizer = optim.Adam(predictor.parameters(), lr=2**(-16))
        a_optimizer = optim.Adam(adversary.parameters(), lr=2**(-16))

        # Set the number of training epochs
        n_epochs = 500

        # Training loop
        start = time.time()
        for epoch in range(n_epochs):
            predictor.train()
            adversary.train()
            train_p_loss = 0.0
            train_a_loss = 0.0
            for inputs, embeds, attribs in train_loader:
                p_optimizer.zero_grad()
                a_optimizer.zero_grad()

                embed_hat = predictor(inputs)
                p_loss = mse_loss(embed_hat, embeds)
                train_p_loss += p_loss.item() * inputs.size(0)

                # If not debiased, only train the predictor using the mse loss
                if not debiased:
                    p_loss.backward()
                    p_optimizer.step()
                    continue

                p_loss.backward(retain_graph=True)

                # Cloning gradients of prediction loss w.r.t. predictor parameters
                dW_LP = [
                    torch.clone(p.grad.detach()) for p in predictor.parameters()
                ]

                p_optimizer.zero_grad()
                a_optimizer.zero_grad()

                attrib_hat = adversary(embed_hat)
                a_loss = mse_loss(attrib_hat, attribs)
                train_a_loss += a_loss.item() * inputs.size(0)
                a_loss.backward()

                # Cloning gradients of adversarial loss w.r.t. predictor parameters
                dW_LA = [
                    torch.clone(p.grad.detach()) for p in predictor.parameters()
                ]

                for i, p in enumerate(predictor.parameters()):
                    # Normalize dW_LA
                    unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + torch.finfo(float).tiny)
                    # Project
                    proj = torch.sum(torch.inner(unit_dW_LA, dW_LP[i]))
                    # Calculate dW according to Zhang et al. (2018)
                    p.grad = dW_LP[i] - (proj * unit_dW_LA) - (dW_LA[i])

                p_optimizer.step()
                a_optimizer.step()
            train_p_loss /= len(train_loader.dataset)
            if debiased: train_a_loss /= len(train_loader.dataset)

            predictor.eval()
            adversary.eval()
            test_p_loss = 0.0
            test_a_loss = 0.0
            with torch.no_grad():
                for inputs, embeds, attribs in test_loader:
                    outputs = predictor(inputs)
                    p_loss = mse_loss(outputs, embeds)
                    test_p_loss += p_loss.item() * inputs.size(0)

                    if debiased:
                        labels = adversary(outputs)
                        a_loss = mse_loss(labels, attribs)
                        test_a_loss += a_loss.item() * inputs.size(0)
            test_p_loss /= len(test_loader.dataset)
            if debiased: test_a_loss /= len(test_loader.dataset)

            if epoch % 20 == 19:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"Train P Loss: {train_p_loss:.4f}, Train A Loss: {train_a_loss:.4f}")
                print(f"Test P Loss: {test_p_loss:.4f}, Test A Loss: {test_a_loss:.4f}")

        end = time.time()
        print(f"Training completed in {end - start} seconds!")

        w = predictor.w.detach().clone().numpy()
        proj = np.dot(w.T, self.gender_vector)
        size = np.linalg.norm(w)
        print(f"Learned w has |w|={size} and <w,g>={proj}.")
