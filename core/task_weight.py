
# Built-in modules
from abc import abstractmethod, ABC

# Third-party modules
import numpy as np

# Custom modules
from core.performance import Performance
from core.epoch_result import EpochResultMTL

class WeightingStrategy(ABC):

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        '''Update the weights based on the current training process.'''
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
        # Apply minimum value constraint by mixing the weights from softmax
        # with a uniform distribution to ensure that no weight is too small.
        # The epsilon parameter controls the strength of this mixing. A higher
        # epsilon value will result in more uniform weights. In this case, the
        # minimum value will be epsilon / number_of_tasks, which ensures that 
        # every task has a non-zero contribution to the overall loss.
        weights = (1.0 - self.epsilon) * weights + self.epsilon / len(weights)

        return weights

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        '''Min-max normalization to [0, 1]'''
        if x.max() - x.min() == 0:
            return np.zeros_like(x)
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
        if not (0 < epsilon < 1):
            raise ValueError("Epsilon must be in the range [0, 1]")

    def update(
        self, 
        epoch_result: EpochResultMTL
    ) -> None:
        '''
        Update the weights based on the current epoch result. The update is
        based on the rank of each task based on the validation AUPRC, with the 
        goal of giving more weight to tasks that are harder (lower AUPRC).

        Parameters
        ----------
        epoch_result : EpochResultMTL
            The result of the current epoch, containing the loss, logits, and
            labels for each task.
        '''
        # Exctract AUPRC
        auprcs = np.array([
            Performance(
                true=epoch_result.labels[idx],
                logits=epoch_result.logits[idx]
            ).auprc
            for idx in range(len(self.weights))
        ])

        # Compute ranks (1 is the best rank, 2 is the second best, etc.)
        ranks = np.argsort(np.argsort(-auprcs)) + 1 

        # Update preweights
        self.preweights += self.learning_rate * (ranks - np.median(ranks))

        # Compute new weights
        self.weights = self._compute_weights()

        # nan check
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Weights contain NaN or Inf values")

class FAMOWeighting(WeightingStrategy):

    def __init__(
        self, 
        number_of_tasks: int,
        learning_rate: float = 0.1,
        lambda_loss: float = 0.3,
        lambda_auprc: float = 0.7,
        epsilon: float = 0.05
    ):  
        # Instance variables
        self.preweights = np.zeros(number_of_tasks)
        self.learning_rate = learning_rate
        self.lambda_loss = lambda_loss
        self.lambda_auprc = lambda_auprc
        self.epsilon = epsilon
        self.weights = self._compute_weights()

        # Validate
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if not (0 < epsilon < 1):
            raise ValueError("Epsilon must be in the range [0, 1]")

    def update(
        self, 
        epoch_result: EpochResultMTL
    ) -> None:
        '''
        Update the weights based on the current epoch result. The update is
        based on the loss and AUPRC of each task, with the goal of giving more
        weight to tasks that are harder (higher loss and lower AUPRC).

        Parameters
        ----------
        epoch_result : EpochResultMTL
            The result of the current epoch, containing the loss, logits, and
            labels for each task.
        '''
        # Exctract loss
        losses = epoch_result.loss

        # Exctract AUPRC
        auprcs = np.array([
            Performance(
                true=epoch_result.labels[idx],
                logits=epoch_result.logits[idx]
            ).auprc
            for idx in range(len(self.weights))
        ])

        # Min-max normalize the loss and AUPRC values to [0, 1]
        # Apply a logarithmic transformation to the loss values to reduce the 
        # impact of large loss values on the weights, and we add a small
        # constant to avoid log(0) which is undefined.
        normalized_losses = self._minmax(np.log(losses + 1e-8))  
        normalized_auprcs = self._minmax(auprcs)

        # Calculate difficulty
        difficulty = (
            self.lambda_loss * normalized_losses + 
            self.lambda_auprc * (1 - normalized_auprcs)
        )

        # Update preweights
        self.preweights += self.learning_rate * (difficulty - np.mean(difficulty))

        # Compute new weights
        self.weights = self._compute_weights()

        # nan check
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Weights contain NaN or Inf values")

if __name__ == "__main__":
    # Try static weighting
    static_weighting = StaticWeighting(weights=[0.1, 0.4, 0.25, 0.25])
    print(f"Static weights: {static_weighting.weights}")

    # Example EpochResultMTL
    epoch_result = EpochResultMTL(
        loss=np.array([0.2, 0.2, 0.4, 0.4]),
        logits=np.array([[-0.1, 0.1], [-0.1, 0.1], [0, 0], [0, 0]]),
        labels=np.array([[0, 1], [0, 1], [0, 1], [0, 1]]),
        identifiers=[]
    )

    # Try FAMO weighting
    famo_weighting = FAMOWeighting(number_of_tasks=4)
    print(f"FAMO weights: {famo_weighting.weights}")
    
    for _ in range(5):
        famo_weighting.update(epoch_result=epoch_result)
        print(f"Updated FAMO weights: {famo_weighting.weights}")

    # Try rank weighting
    rank_weighting = RankWeighting(number_of_tasks=4)
    print(f"Rank weights: {rank_weighting.weights}")
    
    for _ in range(5):
        rank_weighting.update(epoch_result=epoch_result)
        print(f"Updated Rank weights: {rank_weighting.weights}")