"""Evaluation metrics and scikit-learn benchmarking for Naive Bayes classifier."""

from typing import List, Dict, Any, Tuple
import math
from src.naive_bayes import NaiveBayesClassifier

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Calculate classification performance metrics with emphasis on spam detection."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")

    # Convert to standard binary labels: spam = 1 (positive), ham = 0 (negative)
    tp = 0  # True Positive: actual spam, predicted spam
    fp = 0  # False Positive: actual ham, predicted spam (BAD: legitimate email lost)
    tn = 0  # True Negative: actual ham, predicted ham
    fn = 0  # False Negative: actual spam, predicted ham (spam leaked to inbox)

    for yt, yp in zip(y_true, y_pred):
        yt_norm = yt.lower().strip()
        yp_norm = yp.lower().strip()

        if yt_norm == 'spam' and yp_norm == 'spam':
            tp += 1
        elif yt_norm == 'ham' and yp_norm == 'spam':
            fp += 1
        elif yt_norm == 'ham' and yp_norm == 'ham':
            tn += 1
        elif yt_norm == 'spam' and yp_norm == 'ham':
            fn += 1

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Specificity / True Negative Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # False Positive Rate: fraction of legitimate emails mistakenly classified as spam
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        'total_samples': total,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'false_positive_rate': fpr
    }


def format_metrics_report(metrics: Dict[str, Any], title: str = "MODEL EVALUATION METRICS") -> str:
    """Format metrics into a clean text report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"{title}".center(60))
    lines.append("=" * 60)
    lines.append(f"Total Test Samples:        {metrics['total_samples']}")
    lines.append(f"Accuracy:                  {metrics['accuracy'] * 100:.2f}%")
    lines.append(f"Spam Precision:            {metrics['precision'] * 100:.2f}%")
    lines.append(f"Spam Recall:               {metrics['recall'] * 100:.2f}%")
    lines.append(f"Spam F1-Score:             {metrics['f1_score'] * 100:.2f}%")
    lines.append(f"Ham Specificity:           {metrics['specificity'] * 100:.2f}%")
    lines.append(f"False Positive Rate (FPR): {metrics['false_positive_rate'] * 100:.2f}% (legitimate emails marked spam)")
    lines.append("-" * 60)
    lines.append("CONFUSION MATRIX:")
    lines.append(f"                   Predicted Ham      Predicted Spam")
    lines.append(f"  Actual Ham:      {metrics['true_negatives']:<18} {metrics['false_positives']} (False Alarms)")
    lines.append(f"  Actual Spam:     {metrics['false_negatives']:<18} {metrics['true_positives']} (Caught Spam)")
    lines.append("=" * 60)
    return "\n".join(lines)


def benchmark_against_sklearn(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    test_labels: List[str],
    custom_model: NaiveBayesClassifier
) -> Dict[str, Any]:
    """Train scikit-learn's MultinomialNB on the exact same data and compare predictions."""
    if not SKLEARN_AVAILABLE:
        return {'status': 'sklearn not available'}

    # Custom model predictions
    custom_preds = custom_model.predict(test_texts)
    custom_metrics = calculate_metrics(test_labels, custom_preds)

    # Scikit-learn model
    cv = CountVectorizer(lowercase=True, token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    X_train = cv.fit_transform(train_texts)
    y_train = [1 if lbl == 'spam' else 0 for lbl in train_labels]

    sk_nb = MultinomialNB(alpha=custom_model.alpha)
    sk_nb.fit(X_train, y_train)

    X_test = cv.transform(test_texts)
    y_test = [1 if lbl == 'spam' else 0 for lbl in test_labels]

    sk_preds_int = sk_nb.predict(X_test)
    sk_preds = ['spam' if p == 1 else 'ham' for p in sk_preds_int]
    sk_metrics = calculate_metrics(test_labels, sk_preds)

    # Calculate agreement between custom model and sklearn
    agreement = sum(1 for cp, sp in zip(custom_preds, sk_preds) if cp == sp)
    agreement_rate = agreement / len(test_texts) if test_texts else 1.0

    return {
        'status': 'success',
        'custom_metrics': custom_metrics,
        'sklearn_metrics': sk_metrics,
        'prediction_agreement_rate': agreement_rate,
        'agreed_count': agreement,
        'total_tested': len(test_texts)
    }
