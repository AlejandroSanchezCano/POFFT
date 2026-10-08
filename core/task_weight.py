
# Built-in modules
from abc import abstractmethod, ABC

# Third-party modules
import numpy as np

class WeightingStrategy(ABC):

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        pass

    def _compute_weights(self) -> list[float]:
        '''
        Compute the weights based on the preweights using a softmax function,
        ensuring that the weights sum to 1.0. Some tricks are applied to avoid
        numerical instability:
        - Subtracting the maximum preweight from all preweights before applying
            the exponential function to avoid large exponentials.
        - Clipping the preweights to a minimum value of -30 to avoid underflow
            in the exponential function.
        - Applying a minimum value constraint to ensure that no weight is too
            small and every task has a non-zero contribution to the overall 
            loss.
        '''
        # softmax(x) = softmax(x - c) for any constant c, so we can substract
        # the maximum value to avoid numerical instability from large
        # exponentials. This is a common trick when implementing softmax.
        # Now all values are guaranteed to be <= 0.
        preweights = self.preweights - self.preweights.max()
        # Clip preweights to avoid underflow in the exponential function. By 
        # clipping to -30, we ensure that the smallest value of exp(preweights)
        # is exp(-30) ~ 9.357622e-14, which is small enough to be considered
        # negligible in most practical scenarios but without causing numerical
        # issues.
        preweights = np.clip(preweights, -30.0, 0.0)
        # Compute softmax to ensure that the weights sum to 1.0
        exp_preweights = np.exp(preweights)
        weights = exp_preweights / exp_preweights.sum()
        # Apply minimum value constraint
        weights = (1.0 - self.epsilon) * weights + self.epsilon / len(weights)

        return weights

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        '''Min-max normalization to [0, 1]'''
        return (x - x.min()) / (x.max() - x.min())

class StaticWeighting(WeightingStrategy):
    
    def __init__(self, weights: list[float]):
        # Instance variables
        self.weights = weights

        # Validate that the weights sum to 1.0
        if sum(self.weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")

    def update(self, *args, **kwargs) -> None:
        '''Static weighting does not change over time.'''
        pass

class RankWeighting(WeightingStrategy):

    def __init__(
        self, 
        number_of_tasks: int,
        learning_rate: float = 0.1,
        epsilon: float = 0.05
    ):  
        # Instance variables
        self.preweights = np.zeros(number_of_tasks)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weights = self._compute_weights()

        # Validate
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon must be in the range [0, 1]")

    def update(self) -> None:
        pass

class FAMOWeighting(WeightingStrategy):

    def __init__(
        self, 
        number_of_tasks: int,
        learning_rate: float = 0.1,
        learning_rate_loss: float = 0.3,
        learning_rate_auprc: float = 0.7,
        epsilon: float = 0.05
    ):  
        # Instance variables
        self.preweights = np.zeros(number_of_tasks)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weights = self._compute_weights()

        # Validate
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon must be in the range [0, 1]")

    def update(self, epoch_result: 'EpochResult') -> None:
        # Calculate auprc

        # Min-max normalize the loss and AUPRC values to [0, 1]
        normalized_loss = self._minmax(epoch_result.losses)
        normalized_auprc = self._minmax()

if __name__ == "__main__":
    # Try static weighting
    static_weighting = StaticWeighting(weights=[0.1, 0.4, 0.25, 0.25])
    print(f"Static weights: {static_weighting.weights}")

    # Try rank weighting
    rank_weighting = RankWeighting(number_of_tasks=4)
    print(f"Rank weights: {rank_weighting.weights}")