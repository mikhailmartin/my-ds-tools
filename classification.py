from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score


def my_binary_classification_report(
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        *,
        classifier_name: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Предоставляет отчёт по бинарной классификации.

    Визуализирует матрицу ошибок, ROC- и Precision- и Recall- кривые, печатает
    `sklearn.classification_report` и индекс Gini.

    Args:
        y_true: истинные метки.
        y_pred: предсказанные метки.
        y_proba: предсказанные вероятности отнесения к положительному классу.
        classifier_name: имя классификатора.
        figsize: (ширина, высота) рисунка в дюймах.
    """
    if figsize is None:
        figsize = (12.8, 9.6)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[0, 0], colorbar=False)
    axes[0, 0].set(title='Матрица ошибок')
    axes[0, 0].grid()

    sns.histplot(data=y_proba, stat='density', kde=True, ax=axes[0, 1])
    axes[0, 1].set(xlabel='Probability')

    RocCurveDisplay.from_predictions(
        y_true, y_proba, name=classifier_name, ax=axes[1, 0], color='orange')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[1, 0].set(title='ROC-кривая', xlim=(-0.01, 1), ylim=(0, 1.01))

    # Графики зависимости Precision и Recall от порога бинаризации
    precision_recall_plot(y_true, y_proba, ax=axes[1, 1])

    plt.show()

    print(classification_report(y_true, y_pred))

    # Gini_index: https://habr.com/ru/company/ods/blog/350440/
    print(f'Индекс Gini = {2 * roc_auc_score(y_true, y_proba) - 1}')


def my_multiclass_classification_report(
        y_true: pd.Series,
        y_pred: np.ndarray,
        *,
        figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Предоставляет отчёт по мультиклассовой классификации.

    Args:
        y_true: истинные метки.
        y_pred: предсказанные метки.
        figsize: (ширина, высота) рисунка в дюймах.
    """
    if figsize is None:
        figsize = (6.4, 4.8)

    fig, ax = plt.subplots(figsize=figsize)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set(title='Матрица ошибок')

    plt.show()

    print(classification_report(y_true, y_pred))


def precision_recall_plot(
        y_true: pd.Series,
        y_proba: np.ndarray,
        ax: matplotlib.axes.Axes,
        nbin: Optional[int] = 255,
) -> None:
    """
    Рисует на заданном matplotlib.axes.Axes графики зависимости Precision и Recall от порога
    бинаризации.

    Args:
        y_true: истинные метки.
        y_proba: предсказанные вероятности отнесения к положительному классу.
        ax: matplotlib.axes.Axes, на котором следует отрисовать графики.
        nbin: количество бинов для равночастотного биннинга.
    """
    # равночастотный биннинг
    thresholds = np.interp(np.linspace(0, len(y_proba), nbin+1), np.arange(len(y_proba)), np.sort(y_proba))[1: -1]

    threshold_len = len(thresholds)
    precision_scores = np.empty(threshold_len, dtype=float)
    recall_scores = np.empty(threshold_len, dtype=float)

    for i, threshold in enumerate(thresholds):
        y_pred = np.array([1 if proba >= threshold else 0 for proba in y_proba])

        precision_scores[i] = precision_score(y_true, y_pred)
        recall_scores[i] = recall_score(y_true, y_pred)

    ax.plot(thresholds, precision_scores, color='red', label='precision')
    ax.plot(thresholds, recall_scores, color='blue', label='recall')
    ax.set(xlabel='величина порога')
    ax.legend()
