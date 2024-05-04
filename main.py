import argparse

import numpy as np
import torch
from model.complex_gan import ComplexAdversary, ComplexPredictor
from model.simple_gan import SimpleAdversary, SimplePredictor
from trainer.train import Trainer
from utils.loader import load_embeddings_from_np
from utils.preprocess import preprocess_v2, project_onto_subspace, transform_analogies

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Debiasing VL Embeddings")
    parser.add_argument(
        "--embedding_file", type=str, help="Path to the embedding output file"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["simple", "complex"],
        help="Choice between simple or complex model",
    )
    parser.add_argument(
        "--debiased", action="store_true", help="Whether to use debiased embeddings"
    )
    parser.add_argument(
        "--recompute_subspace",
        action="store_true",
        help="Whether to recompute the gender subspace",
    )
    parser.add_argument("--n_epochs", type=int, help="Number of training epochs")

    args = parser.parse_args()
    embedding_file = args.embedding_file
    model_type = args.model_type
    debiased = args.debiased
    recompute_subspace = args.recompute_subspace
    n_epochs = args.n_epochs

    # Load and process data that can be used for training
    vocab, wv, w2i = load_embeddings_from_np("data/embeddings/orig_glove")
    wv = torch.from_numpy(wv).float()
    analogies = preprocess_v2()
    X, y = transform_analogies(wv, w2i, analogies)
    z, gender_subspace = project_onto_subspace(wv, w2i, y)

    # Only predictor can be toggleable between simple and complex as per original Zhang et al. (2018)
    if args.model_type == "simple":
        predictor = SimplePredictor(300)
    elif args.model_type == "complex":
        predictor = ComplexPredictor(300)
    adversary = SimpleAdversary(300)

    # Set up and train the model
    trainer = Trainer(predictor, adversary, gender_subspace, wv, w2i)
    trainer.set_seed(226)
    trainer.train(
        X=X,
        y=y,
        z=z,
        debiased=args.debiased,
        n_epochs=args.n_epochs,
        recompute_subspace=args.recompute_subspace,
    )

    debiased_embeddings = trainer.word_embeds.detach().numpy()

    np.save(f"data/embeddings/{args.embedding_file}", debiased_embeddings)
