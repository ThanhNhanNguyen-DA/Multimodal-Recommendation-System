import numpy as np

class FusionModel:
    def __init__(self, text_weight: float = 0.5, image_weight: float = 0.5):
        self.text_weight = text_weight
        self.image_weight = image_weight

    def fuse_embeddings(self, text_embeddings: np.ndarray, image_embeddings: np.ndarray) -> np.ndarray:
        if text_embeddings.shape[0] != image_embeddings.shape[0]:
            raise ValueError("Text and image embeddings must have the same number of samples.")
        
        # Normalize embeddings
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

        # Fuse embeddings
        fused_embeddings = (self.text_weight * text_embeddings) + (self.image_weight * image_embeddings)
        return fused_embeddings

    def recommend(self, text_embeddings: np.ndarray, image_embeddings: np.ndarray) -> np.ndarray:
        fused_embeddings = self.fuse_embeddings(text_embeddings, image_embeddings)
        return fused_embeddings.argsort(axis=1)[:, ::-1]  # Return indices of top recommendations