"""
===============================================================================
Title:      Performance
Outline:    Performance class to evaluate the performance of a (binary)
            classification model. It calculates the accuracy, balanced 
            accuracy, precision, recall, F1 score, Matthews correlation 
            coefficient, confusion matrix, classification report, ROC curve, 
            confusion matrix, calibration curve...
Author:     Alejandro Sánchez Cano
Date:       29/09/2026
===============================================================================
"""

# Built-in modules
from typing import Literal

# Third-party modules
import torch
import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay

class Performance:

    def __init__(
        self, 
        true: np.ndarray | list, 
        logits: np.ndarray | list | None = None, 
        probs: np.ndarray | list | None= None, 
        threshold: float = 0.5
        ):
        '''
        Class constructor.

        Parameters
        ----------
        true : np.ndarray
            True labels.
        logits : np.ndarray
            Logits from the model.
        probs : np.ndarray
            Predicted probabilities.
        threshold : float, optional
            Threshold to convert logits into probabilities, by default 0.5.
        '''
        self.true = np.array(true)
        if logits is not None:
            logits = np.array(logits)
            self.prob = torch.sigmoid(torch.tensor(logits)).cpu().numpy()
            self.pred = self.prob > threshold
        elif probs is not None:
            probs = np.array(probs)
            self.prob = probs
            self.pred = self.prob > threshold
        else:
            raise ValueError("Either logits or probs must be provided.")

    @property
    def accuracy(self) -> float:
        '''	
        Accuracy metric: (TP + TN) / (TP + TN + FP + FN).

        Returns
        -------
        float
            Accuracy.
        '''
        return sk.accuracy_score(self.true, self.pred)
    
    @property
    def balanced_accuracy(self) -> float:
        '''
        Balanced accuracy metric: (TPR + TNR) / 2.

        Returns
        -------
        float
            Balanced accuracy.
        '''
        return sk.balanced_accuracy_score(self.true, self.pred)
    
    @property
    def precision(self) -> float:
        '''
        Precision metric: TP / (TP + FP).

        Returns
        -------
        float
            Precision.
        '''
        return sk.precision_score(self.true, self.pred)
    
    @property
    def recall(self) -> float:
        '''
        Recall metric: TP / (TP + FN).

        Returns
        -------
        float
            Recall.
        '''
        return sk.recall_score(self.true, self.pred)
    
    @property
    def f1(self) -> float:
        '''
        F1 score metric: 2 * (precision * recall) / (precision + recall).

        Returns
        -------
        float
            F1 score.
        '''
        return sk.f1_score(self.true, self.pred)
    
    @property
    def mcc(self) -> float:
        '''
        Matthews correlation coefficient metric: 
        (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).

        Returns
        -------
        float
            Matthews correlation coefficient.
        '''
        return sk.matthews_corrcoef(self.true, self.pred)
    
    @property
    def confusion_matrix(self) -> np.ndarray:
        '''
        Confusion matrix metric.

        Returns
        -------
        np.ndarray
            Confusion matrix.
        '''
        return sk.confusion_matrix(self.true, self.pred)
    
    def plot_confusion_matrix(self) -> None:
        '''Plot the confusion matrix.'''
        cm = sk.confusion_matrix(self.true, self.pred)
        cm = sk.ConfusionMatrixDisplay(cm, display_labels = ['-', '+'])
        cm.plot()
        plt.savefig('confusion_matrix.png')
        plt.clf()

    @property
    def fpr(self) -> np.ndarray:
        '''
        Compute the False Positive Rate (FPR) for the ROC.

        Returns
        -------
        np.ndarray
            False Positive Rate.
        '''
        fpr, _, _ = sk.roc_curve(self.true, self.prob)
        return fpr

    @property
    def tpr(self) -> np.ndarray:
        '''
        Compute the True Positive Rate (TPR) for the ROC.

        Returns
        -------
        np.ndarray
            True Positive Rate.
        '''
        _, tpr, _ = sk.roc_curve(self.true, self.prob)
        return tpr

    @property
    def auc(self) -> float:
        '''
        Compute the Area Under the Curve (AUC) for the ROC.

        Returns
        -------
        float
            AUC score.
        '''
        fpr, tpr, _ = sk.roc_curve(self.true, self.prob)
        return sk.auc(fpr, tpr)

    @property
    def classification_report(self) -> str:
        '''
        Classification report metric.

        Returns
        -------
        str
            Classification report.
        '''
        return sk.classification_report(self.true, self.pred)
    
    def plot_roc_curve(self) -> None:
        '''Plot the ROC curve.'''
        fpr, tpr, thresholds = sk.roc_curve(self.true, self.prob)
        roc_auc = sk.auc(fpr, tpr)
        optimal_distance_threshold_idx = np.argmin(np.sqrt((1-tpr)**2 + fpr**2))
        optimal_youden_threshold_idx = np.argmax(tpr - fpr)
        sk.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        #plt.plot(fpr[optimal_distance_threshold_idx], tpr[optimal_distance_threshold_idx], 'rs', label=f'Optimal Threshold (Min Distance) = {thresholds[optimal_distance_threshold_idx]:.2f}')
        #plt.plot(fpr[optimal_youden_threshold_idx], tpr[optimal_youden_threshold_idx], 'go', label=f'Optimal Threshold (Youden Index) = {thresholds[optimal_youden_threshold_idx]:.2f}')
        plt.plot([0, 1], [0, 1], color='k', linestyle='--')  # Random classifier line
        #plt.legend()
        plt.savefig('roc_curve.png')
        plt.clf()
    
    def plot_calibration_curve(
            self, 
            n_bins: int, 
            strategy: Literal['uniform', 'qualtile'], 
            model_name: str
            ) -> None:
        '''
        Plot the calibration curve.

        Parameters
        ----------
        n_bins : int
            Number of bins.
        strategy : {'uniform', 'quantile'}
            Strategy to create the bins.
        model_name : str
            Model name.
        '''
        CalibrationDisplay.from_predictions(
            y_true=self.true, 
            y_prob=self.prob, 
            n_bins=n_bins, 
            strategy=strategy, 
            name=model_name).plot()
        plt.savefig('calibration_curve.png')
        plt.clf()

    def plot_pr_curve(self) -> None:
        '''Plot the Precision-Recall curve.'''
        precision, recall, _ = sk.precision_recall_curve(self.true, self.prob)
        sk.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        plt.axhline(y=np.mean(self.true), color='k', linestyle='--')  # Random classifier line
        plt.savefig('pr_curve.png')
        plt.clf()

    @property
    def auprc(self) -> float:
        '''
        Compute the Area Under the Precision-Recall Curve (AUPRC).

        Returns
        -------
        float
            AUPRC score.
        '''
        auprc = sk.average_precision_score(self.true, self.prob)
        return auprc

if __name__ == "__main__":
    true = np.array(np.random.randint(0, 2, size=1000))
    probs = np.array(np.random.rand(1000))
    p = Performance(true, probs = probs)
    p.plot_roc_curve()
    p.plot_pr_curve()
