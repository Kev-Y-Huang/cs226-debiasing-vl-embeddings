from model.simple_gan import SimpleAdversary, SimplePredictor
from trainer.train import Trainer
from utils.loader import load_embeddings_from_np

if __name__=='__main__':
    vocab, wv, w2i = load_embeddings_from_np("data/embeddings/orig_w2v")
    trainer = Trainer(wv, w2i)
    X, y, a = trainer.preprocess()

    predictor = SimplePredictor(300)
    adversary = SimpleAdversary(300)

    trainer.train(
        X=X,
        y=y,
        a=a,
        predictor=predictor,
        adversary=adversary,
        debiased=False
    )