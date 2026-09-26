
class ESM2Embedding:
    def __init__(self, embedding: 'torch.Tensor'):
        self.embedding = embedding

    def __repr__(self):
        shape = list(self.embedding.shape)
        return f"{type(self).__name__}({shape})"

    def remove_special_tokens(self) -> 'torch.Tensor':
        if self.embedding.dim() == 3:
            return self.embedding[:, 1:-1, :]
        elif self.embedding.dim() == 2:
            return self.embedding[1:-1, :]
        else:
            raise NotImplementedError("Only implemented for 2D/3D embeddings.")
    
    def mean_pool(self) -> 'torch.Tensor':
        return self.embedding.mean(dim=-2)

    def concatenate(self, other: 'ESM2Embedding') -> 'ESM2Embedding':
        concatenated = torch.cat([self.embedding, other.embedding], dim=-1)
        return ESM2Embedding(concatenated)

if __name__ == "__main__":
    import torch
    embedding = torch.rand(3, 10)  # Example embedding tensor
    esm2_embedding = ESM2Embedding(embedding)
    print(esm2_embedding) 