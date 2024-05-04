import numpy as np
import torch
from model.complex_gan import ComplexAdversary, ComplexPredictor
from model.simple_gan import SimpleAdversary, SimplePredictor
from tqdm import tqdm
from trainer.train import Trainer
from utils.loader import load_embeddings_from_np
from utils.preprocess import (preprocess, preprocess_v2, project_onto_subspace,
                              transform_analogies)

if __name__ == "__main__":
    vocab, wv, w2i = load_embeddings_from_np("data/embeddings/orig_glove")
    analogies = preprocess_v2()
    X, y = transform_analogies(wv, w2i, analogies)
    z, gender_subspace = project_onto_subspace(wv, w2i, y)

    predictor = SimplePredictor(300)
    adversary = SimpleAdversary(300)
    trainer = Trainer(predictor, adversary, gender_subspace)
    trainer.set_seed(42)

    trainer.train(X=X, y=y, z=z, debiased=True)
