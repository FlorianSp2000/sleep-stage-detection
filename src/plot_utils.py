import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_hypnogram(y_true, y_pred, size=500):

    time = np.arange(len(y_true[:size]))

    sleep_stages = {0: "Wake", 1: "N1 sleep", 2: "N2 sleep", 3: "N3 sleep", 4: "REM sleep"}

    plt.figure(figsize=(10, 5))
    plt.plot(time, y_pred[:size], drawstyle='steps-post', color='b')
    plt.plot(time, y_true[:size], drawstyle='steps-post', color='r')

    plt.yticks(np.arange(len(sleep_stages)), list(sleep_stages.values()))
    plt.xlabel('Intervals')
    plt.ylabel('Sleep Stage')
    plt.title('Hypnogram')

    plt.grid(True)
    plt.show()


def plot_confusion_matrix(conf_matrix, classes, figsize=(8, 6), cmap='Blues', normalize=None):
    """
    Plots a confusion matrix with predicted vs ground truth labels.
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix (output of confusion_matrix function).
        classes (list): List of class labels in the same order as used in the confusion matrix.
        figsize (tuple): Figure size for the plot.
        cmap (str): Color map for the heatmap.
    """
    plt.figure(figsize=figsize)
    fmt = '.2f' if normalize else 'd'
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=classes, yticklabels=classes, cbar=False)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('Ground Truth Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_class_agreement(metrics):
    """
    Plot a stacked bar chart showing agreement and disagreement percentages for each class.

    Parameters:
    metrics (dict): A dictionary containing class-specific metrics.
                    Expected structure:
                    {
                        'class_metrics': {
                            class_label: {
                                'agreement_percentage': float,
                                ...
                            },
                            ...
                        }
                    }
    """
    class_metrics = metrics['class_metrics']
    classes = sorted(class_metrics.keys())
    agreement_percentages = [class_metrics[cls]['agreement_percentage'] for cls in classes]
    disagreement_percentages = [100 - perc for perc in agreement_percentages]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(classes))
    width = 0.6
    
    # Modern colors
    agreement_color = '#2ecc71'  # A more vibrant green
    disagreement_color = '#e74c3c'  # A more vibrant red
    
    bars1 = ax.bar(x, agreement_percentages, width, label='Agreement', color=agreement_color, alpha=0.8)
    bars2 = ax.bar(x, disagreement_percentages, width, bottom=agreement_percentages, label='Disagreement', color=disagreement_color, alpha=0.8)
    
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Sleep Stage')
    ax.set_title('Agreement/Disagreement by Sleep Stage')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    
    # Add percentage labels on bars
    def add_percentage_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_percentage_labels(bars1)
    
    plt.tight_layout()
    plt.show()