from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, recall_score

def compute_average_precision_score(ground_truth, predictions, num_classes):
    # binarize the labels for groundtruth and predictions
    ground_truth_binarized = label_binarize(ground_truth, classes=list(range(0, num_classes)))
    predictions_binarized = label_binarize(predictions, classes=list(range(0, num_classes)))

    average_precision = average_precision_score(predictions_binarized, ground_truth_binarized, average="micro")
    return average_precision


def compute_average_recall(ground_truth, predictions, num_classes):
    # binarize the labels for groundtruth and predictions
    ground_truth_binarized = label_binarize(ground_truth, classes=list(range(0, num_classes)))
    predictions_binarized = label_binarize(predictions, classes=list(range(0, num_classes)))

    average_recall = recall_score(predictions_binarized, ground_truth_binarized, average="micro")
    return average_recall