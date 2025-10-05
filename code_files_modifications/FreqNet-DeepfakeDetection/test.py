import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.freqnet import freqnet
from options.test_options import TestOptions
import numpy as np
import random
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
                             precision_recall_curve)

# =========================
# Auxiliary function for Expected Calibration Error (ECE)
# =========================
def expected_calibration_error(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        idx = bin_ids == i
        if np.sum(idx) == 0:
            continue
        acc_bin = np.mean(y_true[idx])
        conf_bin = np.mean(y_prob[idx])
        ece += (np.sum(idx) / len(y_true)) * abs(acc_bin - conf_bin)
    return ece

# =========================
# Function to compute extra metrics
# =========================
def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {}
    metrics['n_samples'] = int(len(y_true))
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics['roc_auc'] = None
    try:
        metrics['pr_auc'] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics['pr_auc'] = None
    metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else None
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)})
    metrics['ece'] = float(expected_calibration_error(y_true, y_score, n_bins=15))

    return metrics

# =========================
# Function to save metrics to CSV
# =========================
def save_metrics_csv(metrics_list, csv_path):
    df = pd.DataFrame(metrics_list)
    df.to_csv(csv_path, index=False)

# =========================
# Test dataset configuration
# =========================
DetectionTests = {

    'GANGen-Detection': {
        'dataroot': '/content/FreqNet-DeepfakeDetection/dataset/GANGen-Detection',
        'no_resize': True,
        'no_crop': True,
    },
}

# =========================
# Load options and model
# =========================
opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

model = freqnet(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()

# =========================
# Main testing loop
# =========================
for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs, aps = [], []
    all_metrics = []
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    for v_id, val in enumerate(sorted(os.listdir(dataroot))):
        opt.dataroot = f'{dataroot}/{val}'
        opt.classes = ''
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop = DetectionTests[testSet]['no_crop']

        # Now validate should also return y_true and y_score
        acc, ap, precision, recall, f1, auc, cm, y_true, y_score = validate(model, opt)


        accs.append(acc)
        aps.append(ap)

        # Compute extra metrics
        metrics = compute_metrics(y_true, y_score, threshold=0.5)
        metrics.update({
            'dataset': testSet,
            'subfolder': val,
            'acc_from_validate': acc,
            'ap_from_validate': ap
        })
        all_metrics.append(metrics)

        print(f"({v_id} {val:12}) acc: {acc*100:.1f}; ap: {ap*100:.1f}; f1: {metrics['f1']*100:.1f}; auc: {metrics['roc_auc']:.3f}")

    # Save metrics per subfolder
    csv_path = f'{testSet}_metrics.csv'
    save_metrics_csv(all_metrics, csv_path)
    print(f"Metrics saved to {csv_path}")

    # Print overall average
    print(f"({v_id+1} {'Mean':10}) acc: {np.mean(accs)*100:.1f}; ap: {np.mean(aps)*100:.1f}")
    print('*' * 25)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
