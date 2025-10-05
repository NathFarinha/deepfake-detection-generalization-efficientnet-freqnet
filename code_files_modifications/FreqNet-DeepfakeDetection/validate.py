import torch
import numpy as np
from networks.freqnet import freqnet
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            # Predict scores, apply sigmoid to get probabilities (0 to 1), flatten, and convert to list
            preds = model(in_tens).sigmoid().flatten().tolist()
            y_pred.extend(preds)
            # Flatten labels and convert to list
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Fixed threshold for binarizing predictions
    threshold = 0.5
    y_pred_binary = (y_pred > threshold).astype(int)

    # Metrics
    # Accuracy on Real (True Negative Rate / Specificity)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred_binary[y_true == 0])
    # Accuracy on Fake (True Positive Rate / Recall)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred_binary[y_true == 1])
    acc = accuracy_score(y_true, y_pred_binary)
    ap = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred_binary)

    # Return all metrics, including the ground truth and prediction scores
    return acc, ap, r_acc, f_acc, f1, auc, cm, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = freqnet(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    # Load state_dict from the 'model' key (assuming the checkpoint structure)
    model.load_state_dict(state_dict['model']) 
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, f1, auc, cm, y_true, y_pred = validate(model, opt)

    print(f"Accuracy: {acc:.4f}")
    print(f"Average Precision (AP): {avg_precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Accuracy (Real images): {r_acc:.4f}")
    print(f"Accuracy (Fake images): {f_acc:.4f}")
    print("Confusion Matrix:\n", cm)
