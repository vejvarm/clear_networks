from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, GRU, Dense


class GRUModel(Model):
    EmbeddingSize = int
    GRUSize = int
    DenseSize = int
    Sizes = Tuple[EmbeddingSize, GRUSize, DenseSize]

    def __init__(self, vocab_size: int, sizes: Sizes):
        super(GRUModel, self).__init__()

        embedding_size, gru_size, dense_size = sizes

        if not embedding_size:
            embedding_size = 64
        if not gru_size:
            gru_size = 64
        if not dense_size:
            dense_size = 64

        self.embedding = Embedding(vocab_size, embedding_size, mask_zero=True)
        self.gru = GRU(gru_size)
        self.dense = Dense(dense_size)
        self.class_out = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.embedding(inputs)
        mask = self.embedding.compute_mask(inputs)
        x = self.gru(x, mask=mask)
        x = self.dense(x)
        return self.class_out(x)
