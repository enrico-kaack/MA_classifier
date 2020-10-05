from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score, f1_score
import os

def plot_confusion_matrix(task_id, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    print(cm_normalized)
    print('Confusion matrix, without normalization')
    print(cm)
    results = {"cm": cm, "cm_normalized": cm_normalized}



    if normalize:
        cm = cm_normalized


    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

    #check if path exists, otherwise create
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/CM_{task_id}.svg")
    return results

def evaluate_predictions(metric_prefix, predictions, probabilities, labels):
    results = {}
    
    results[metric_prefix + '_recall'] = recall_score(labels, predictions)
    results[metric_prefix + '_precision'] = precision_score(labels, predictions)
    results[metric_prefix + '_roc'] = roc_auc_score(labels, probabilities)
    results[metric_prefix + '_f1'] = f1_score(labels, predictions)
    return results

def evaluate_model(task_id, predictions, probs, test_labels, train_predictions, train_probs, train_labels, train_dev_predictions, train_dev_probs, train_dev_labels, only_test=False):
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    results['f1'] = f1_score(test_labels, predictions)

    if not only_test:
        results['train_recall'] = recall_score(train_labels, train_predictions)
        results['train_precision'] = precision_score(train_labels, train_predictions)
        results['train_roc'] = roc_auc_score(train_labels, train_probs)
        results['train_f1'] = f1_score(train_labels, train_predictions)

        results['train_dev_recall'] = recall_score(train_dev_labels, train_dev_predictions)
        results['train_dev_precision'] = precision_score(train_dev_labels, train_dev_predictions)
        results['train_dev_roc'] = roc_auc_score(train_dev_labels, train_dev_probs)
        results['train_dev_f1'] = f1_score(train_dev_labels, train_dev_predictions)

    print("Test Accuracy",  accuracy_score(test_labels, predictions))
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5

    if not only_test:

        train_results = {}
        train_results['recall'] = recall_score(train_labels, train_predictions)
        train_results['precision'] = precision_score(train_labels, train_predictions)
        train_results['roc'] = roc_auc_score(train_labels, train_probs)

    
        for metric in ['recall', 'precision', 'roc']:
            print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    else:
        for metric in ['recall', 'precision', 'roc']:
            print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    #plt.tight_layout(rect=(0.05, 0.05, 1, 0.95))
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.savefig(f"results/plots/ROC_{task_id}.svg");

    return results