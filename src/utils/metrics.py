def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()

def precision_score(y_true, y_pred):
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    predicted_positives = (y_pred == 1).sum()
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def recall_score(y_true, y_pred):
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    actual_positives = (y_true == 1).sum()
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def mean_average_precision(y_true, y_pred_scores):
    sorted_indices = np.argsort(y_pred_scores)[::-1]
    sorted_y_true = y_true[sorted_indices]
    cumulative_hits = np.cumsum(sorted_y_true)
    average_precisions = cumulative_hits / (np.arange(len(y_true)) + 1)
    return np.mean(average_precisions) if len(y_true) > 0 else 0.0