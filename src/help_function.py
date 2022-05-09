import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import logging as log
import os


def train_classifier(classifier, classifier_name: str, train_features, train_labels, test_features):
    log.info("training classifier {0}".format(classifier_name))
    start_time = time.time()
    log.info("training classifier {0}: start time = {1}".format(classifier_name, start_time))
    trained_classifier = classifier.fit(train_features, train_labels)
    predicted_labels = classifier.predict(test_features)
    end_time = time.time()
    log.info("trained classifier {0}: end time = {1}".format(classifier_name, end_time))
    model_run_time = round(end_time - start_time, 2)
    log.info("trained classifier {0}: time elapsed = {1} seconds (2 d.p.)".format(classifier_name, model_run_time))
    return trained_classifier, predicted_labels, model_run_time


def compute_evaluation_metrics(classifier_name: str, test_labels, predicted_labels):
    true_negatives, false_positives, false_negatives, true_positives = \
        confusion_matrix(test_labels, predicted_labels).ravel()
    accuracy = metrics.accuracy_score(predicted_labels, test_labels)
    precision = metrics.precision_score(predicted_labels, test_labels)
    recall = metrics.recall_score(predicted_labels, test_labels)
    specificity = true_negatives / (true_negatives + false_positives)
    mcc = metrics.matthews_corrcoef(predicted_labels, test_labels)
    roc_auc = metrics.roc_auc_score(predicted_labels, test_labels)
    return {
        "Model": classifier_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Matthews correlation coefficient": mcc,
        "ROC-AUC score": roc_auc
    }


def print_confusion_matrix(results_folder, dataset_name: str, classifier_name: str, predicted_labels, actual_labels):
    assert classifier_name
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(actual_labels, predicted_labels)
    hm = sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", label="small")
    hm.set_xticklabels(labels=["Slight", "Fatal", "Serious"], rotation=0)
    hm.set_yticklabels(labels=["Slight", "Fatal", "Serious"], rotation=90)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion Matrix: {0}".format(classifier_name))
    assert os.path.exists(results_folder)
    figure_name = "{0}_{1}_cm.png".format(dataset_name, classifier_name.replace(" ", "_"))
    figure_path = os.path.abspath(os.path.join(results_folder, figure_name))
    plt.savefig(figure_path, bbox_inches="tight")


def print_all_roc_curves(
        classifiers, results_folder, dataset_name: str, test_features, test_labels):
    predict_proba_lr = classifiers[0].predict_proba(test_features)[::, 1]
    fpr1, tpr1, _ = metrics.roc_curve(test_labels, predict_proba_lr)
    auc1 = metrics.roc_auc_score(test_labels, predict_proba_lr)

    predict_proba_lsvm = classifiers[1].predict_proba(test_features)[::, 1]
    fpr2, tpr2, _ = metrics.roc_curve(test_labels, predict_proba_lsvm)
    auc2 = metrics.roc_auc_score(test_labels, predict_proba_lsvm)

    predict_proba_rbf = classifiers[2].predict_proba(test_features)[::, 1]
    fpr3, tpr3, _ = metrics.roc_curve(test_labels, predict_proba_rbf)
    auc3 = metrics.roc_auc_score(test_labels, predict_proba_rbf)

    predict_proba_poly = classifiers[3].predict_proba(test_features)[::, 1]
    fpr4, tpr4, _ = metrics.roc_curve(test_labels, predict_proba_poly)
    auc4 = metrics.roc_auc_score(test_labels, predict_proba_poly)

    predict_proba_nb = classifiers[4].predict_proba(test_features)[::, 1]
    fpr5, tpr5, _ = metrics.roc_curve(test_labels, predict_proba_nb)
    auc5 = metrics.roc_auc_score(test_labels, predict_proba_nb)

    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr1, tpr1, label="Logistic Regression, auc=" + str(round(auc1, 2)))
    plt.plot(fpr2, tpr2, label="Feed Forward Neural Network, auc=" + str(round(auc2, 2)))
    plt.plot(fpr3, tpr3, label="Random Forest, auc=" + str(round(auc3, 2)))
    plt.legend(loc=4, title='Models', facecolor='white')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC', size=15)
    assert os.path.exists(results_folder)
    figure_name = "{0}_roc.png".format(dataset_name)
    figure_path = os.path.abspath(os.path.join(results_folder, figure_name))
    plt.savefig(figure_path, bbox_inches="tight")
