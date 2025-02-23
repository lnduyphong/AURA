import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class DeterministicAgent:
    def __init__(self, dataset: pd.DataFrame, 
                 probs_1: np.ndarray, 
                 probs_2: np.ndarray, 
                 probs_3: np.ndarray,
                 threshold: float = 0.7,
                 similarity_weight: float = 0.5):
        
        self.dataset = dataset
        self.probs = [probs_1, probs_2, probs_3]
        self.threshold = threshold
        self.similarity_weight = similarity_weight
        self._validate_inputs()

    def _validate_inputs(self):
        """Ensure consistent input dimensions"""
        n_samples = len(self.dataset)
        if not all(p.shape[0] == n_samples for p in self.probs):
            raise ValueError("All inputs must have the same number of samples")
            
        if len(self.probs[0].shape) != 2:
            raise ValueError("Probability matrices should be 2D (samples × classes)")

    def _compute_consensus(self):
        """Calculate weighted consensus using model confidence"""
        # Calculate weights based on model confidence
        weights = [np.max(p, axis=1) for p in self.probs]
        total_weights = sum(weights)
        
        # Weighted probability average
        weighted_probs = sum(w[:, None] * p for w, p in zip(weights, self.probs))
        weighted_probs /= total_weights[:, None]
        
        self.consensus = np.argmax(weighted_probs, axis=1)
        self.confidence = np.max(weighted_probs, axis=1)
        return self.consensus, self.confidence

    def _embedding_analysis(self):
        """Analyze embedding consistency with labels"""
        embeddings = self.dataset.iloc[:, :-1].values
        current_labels = self.dataset.iloc[:, -1].values
        
        # Calculate class prototypes
        self.prototypes = {}
        unique_labels = np.unique(np.concatenate([current_labels, self.consensus]))
        for label in unique_labels:
            mask = (current_labels == label)
            if np.sum(mask) > 0:
                self.prototypes[label] = np.mean(embeddings[mask], axis=0)
            else:
                self.prototypes[label] = np.zeros(embeddings.shape[1])

        # Calculate similarity scores
        self.current_similarity = np.array([cosine_similarity([e], [self.prototypes[l]])[0,0] 
                                          for e, l in zip(embeddings, current_labels)])
        
        self.consensus_similarity = np.array([cosine_similarity([e], [self.prototypes[c]])[0,0] 
                                            for e, c in zip(embeddings, self.consensus)])

    def _compute_adjusted_confidence(self):
        """Combine model confidence with embedding consistency"""
        similarity_ratio = self.consensus_similarity / (self.current_similarity + 1e-8)
        return self.confidence * (1 + self.similarity_weight * (similarity_ratio - 1))

    def make_decision(self):
        """Main decision-making function with layered checks"""
        # Step 1: Compute basic consensus
        consensus, confidence = self._compute_consensus()
        
        # Step 2: Embedding space analysis
        self._embedding_analysis()
        
        # Step 3: Combine confidence sources
        adjusted_confidence = self._compute_adjusted_confidence()
        
        # Step 4: Adaptive thresholding
        current_labels = self.dataset.iloc[:, -1].values
        update_mask = (
            (adjusted_confidence >= self.threshold) &
            (consensus != current_labels) &
            (self.consensus_similarity > self.current_similarity)
        )

        # Create confidence-based tiers
        high_confidence = adjusted_confidence >= self.threshold * 1.2
        medium_confidence = adjusted_confidence >= self.threshold
        
        # Final decision matrix
        final_labels = np.where(
            high_confidence & (consensus != current_labels),
            consensus,
            np.where(
                medium_confidence & (self.consensus_similarity > 0.8),
                consensus,
                current_labels
            )
        )

        self.dataset.iloc[:, -1] = final_labels
        return self.dataset

    def get_questionable_samples(self):
        """Identify samples needing human review"""
        adjusted_confidence = self._compute_adjusted_confidence()
        return self.dataset[
            (adjusted_confidence < self.threshold) &
            (self.consensus != self.dataset.iloc[:, -1])
        ]