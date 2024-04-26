from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# from utils.metrics import *
# from utils.dataloader import dataloader
# from model.TF2.AdversarialDebiasing import AdversarialDebiasing as AdversarialDebiasingTF2
# from model.Torch.AdversarialDebiasing import AdversarialDebiasing as AdversarialDebiasingTorch
# import matplotlib.pyplot as plt
import time
import sys

from tqdm import tqdm
import numpy as np
import requests
from source.loader import load_embeddings_from_np
import torch
import torch.nn as nn
import torch.optim as optim
from model.simple_gan import SimpleGAN


def preprocess(wv, w2i):
    """
    Preprocesses the analogy data to produce input, output, and protected attribute data.

    Args:
        wv: Word vectors
        w2i: Word to index mapping

    Returns:
        X: Input data as embeddings of the first three words in each analogy
        y: Output data as embeddings of the fourth word in each analogy
        a: Protected attribute data as the cosine similarity of each output embedding
    """
    # Load analogy data
    url = 'http://download.tensorflow.org/data/questions-words.txt'
    # Family category includes gender dynamics
    categories = ['family']
    r = requests.get(url, allow_redirects=False)
    lines = r.text.split('\n')
    analogies = []
    valid_category = False
    for line in lines:
        sp = line.split(' ')
        # Start of category will be preceded by the line ": family"
        if len(sp) == 2:
            valid_category = sp[1] in categories
        # Each analogy will be formatted as "a b c d" where a:b::c:d
        elif len(sp) == 4 and valid_category:
            analogies.append(sp)

    print(f"{len(analogies)} analogies!")

    X = []
    y = []
    for analogy in tqdm(analogies):
        x = []
        for word in analogy[:-1]:
            x.append(wv[w2i[word]])
        X.append(x)
        y.append(wv[w2i[analogy[-1]]])
    X = np.array(X)
    y = np.array(y)

    # Calculate "gender direction"
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
    m = np.array([wv[w2i[wf]] - wv[w2i[wm]] for wf, wm in pairs])
    m = np.cov(np.array(m).T)
    evals, evecs = np.linalg.eig(m)
    dir = np.real(evecs[:, np.argmax(evals)])
    gender_direction = dir / np.linalg.norm(dir) # normalized

    # Calculate sensitive feature
    a = np.array([np.dot(y_, gender_direction) for y_ in y])

    return X, y, a


def train(X, y, a, generator, discriminator, debiased=True):
    # Split data into train and test
    x_train, x_test, y_train, y_test, a_train, a_test = train_test_split(X,
                                                        y,
                                                        a,
                                                        test_size=0.1,
                                                        random_state=42,
                                                        stratify=a)

    # Define the loss function and optimizer
    g_criterion = nn.MSELoss()
    d_criterion = nn.BCELoss()

    g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    # Convert the data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    a_train_tensor = torch.tensor(a_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    a_test_tensor = torch.tensor(a_test, dtype=torch.float32)

    # Create a PyTorch DataLoader for training and testing data
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor, a_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor, a_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set the number of training epochs
    n_epochs = 100

    # Training loop
    start = time.time()
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        train_g_loss = 0.0
        train_d_loss = 0.0
        train_correct = 0
        for inputs, embeds, attribs in train_loader:
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # Get outputs from each network
            outputs = generator(inputs)
            labels = discriminator(outputs)

            # Calculate losses and backpropagate for generator
            g_loss = g_criterion(outputs, embeds)
            g_loss.backward()
            g_optimizer.step()
            train_g_loss += g_loss.item() * inputs.size(0)

            predicted = torch.argmax(labels.data, 1)
            train_correct += (predicted == labels).sum().item()

            if debiased:
                # Calculate losses and backpropagate for discriminator
                d_loss = d_criterion(labels, attribs)
                d_loss.backward()
                d_optimizer.step()
                train_d_loss += d_loss.item() * inputs.size(0)
        train_g_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        if debiased: train_d_loss /= len(train_loader.dataset)

        generator.eval()
        discriminator.eval()
        test_g_loss = 0.0
        test_d_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            for inputs, embeds, attribs in test_loader:
                outputs = generator(inputs)
                g_loss = g_criterion(outputs, embeds)
                test_g_loss += g_loss.item() * inputs.size(0)

                predicted = torch.argmax(labels.data, 1)
                test_correct += (predicted == labels).sum().item()

                if debiased:
                    labels = discriminator(outputs)
                    d_loss = d_criterion(labels, attribs)
                    test_d_loss += d_loss.item() * inputs.size(0)
        test_g_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / len(test_loader.dataset)
        if debiased: test_d_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train G Loss: {train_g_loss:.4f}, Train D Loss: {train_d_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test G Loss: {test_g_loss:.4f}, Train D Loss: {test_d_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    end = time.time()
    print(f"Training completed in {end - start} seconds!")


    # print("\nTrain Results\n")
    # y_pred = pipeline.predict(x_train)
    # ACC = accuracy_score(y_train[classification], y_pred)
    # DEO = DifferenceEqualOpportunity(y_pred, y_train, sensitive_feature, classification, 1, 0, [0, 1])
    # DAO = DifferenceAverageOdds(y_pred, y_train, sensitive_feature, classification, 1, 0, [0, 1])
    # print(f'\nTrain Acc: {ACC}, \nDiff. Equal Opportunity: {DEO}, \nDiff. in Average Odds: {DAO}')

    # start = time.time()
    # print("\nTest Results\n")
    # y_pred = pipeline.predict(x_test)
    # ACC = accuracy_score(y_test[classification], y_pred)
    # DEO = DifferenceEqualOpportunity(y_pred, y_test, sensitive_feature, classification, 1, 0, [0, 1])
    # DAO = DifferenceAverageOdds(y_pred, y_test, sensitive_feature, classification, 1, 0, [0, 1])
    # print(f'\nTest Acc: {ACC}, \nDiff. Equal Opportunity: {DEO}, \nDiff. in Average Odds: {DAO}')
    # end = time.time()
    # total = end - start
    # print(f"{CLF_type} with {Backend} Backend Inference completed in {total} seconds!")


if __name__=='__main__':
    info = sys.argv[1].split('_')
    dataframe = info[0]
    sensitive_feature = info[1]
    Backend = info[2]

    vocab, wv, w2i = load_embeddings_from_np("../data/embeddings/orig_w2v")
    X, y, a = preprocess(wv, w2i)

    model = SimpleGAN(300)

    train(
        X=X,
        y=y,
        a=a,
        generator=model.generator,
        discriminator=model.discriminator,
        debiased=False
    )
    train(
        X=X,
        y=y,
        a=a,
        generator=model.generator,
        discriminator=model.discriminator,
        debiased=True
    )