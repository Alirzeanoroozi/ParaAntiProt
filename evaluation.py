from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve, average_precision_score, \
    recall_score, precision_score, f1_score, confusion_matrix
# from plot import plot_roc_curve
import numpy as np


def youden_j_stat(fpr, tpr, thresholds):
    j_ordered = sorted(zip(tpr - fpr, thresholds))
    return 1. if j_ordered[-1][1] > 1 else j_ordered[-1][1]


def compute_classifier_metrics(probs, labels, lengths, epoch, file_name, threshold=None, cv=None, config=None):
    probs = probs.detach()

    if config['dataset'] == 'nano':
        efective_labels = sum([lbl[:l].tolist() for lbl, l in zip(labels, lengths)], [])
        efective_probs = sum([p[:l].tolist() for p, l in zip(probs, lengths)], [])
        jstat = youden_j_stat(*roc_curve(efective_labels, efective_probs))
    else:
        jstats = []
        for lbl, p, l in zip(labels, probs, lengths):
            jstats.append(youden_j_stat(*roc_curve(lbl[:l], p[:l])))
        jstat_scores = np.array(jstats)
        jstat = np.mean(jstat_scores)

    if threshold is None:
        threshold = jstat

    if config['dataset'] == 'nano':
        print("lbl", *[1 if x == 1.0 else 0 for x in efective_labels])
        auc = roc_auc_score(efective_labels, efective_probs)
        aupr = average_precision_score(efective_labels, efective_probs, pos_label=1.0)
        l_pred = (efective_probs > threshold).astype(int)
        print("pre", *l_pred.tolist())
        mcorr = matthews_corrcoef(efective_labels, l_pred)
        rec = recall_score(efective_labels, l_pred)
        prec = precision_score(efective_labels, l_pred)
        fsc = f1_score(efective_labels, l_pred)
    else:
        matrices = []
        aucs = []
        aupr = []
        mcorrs = []
        for lbl, p, l in zip(labels, probs, lengths):
            # print("lbl", *[1 if x == 1.0 else 0 for x in lbl[:l].tolist()])
            try:
                aucs.append(roc_auc_score(lbl[:l], p[:l]))
            except:
                pass
            aupr.append(average_precision_score(lbl[:l], p[:l], pos_label=1.0))
            l_pred = (p[:l] > threshold).numpy().astype(int)
            # print("pre", *l_pred.tolist())
            matrices.append(confusion_matrix(lbl[:l], l_pred, labels=[0, 1]))
            mcorrs.append(matthews_corrcoef(lbl[:l], l_pred))

        matrices = np.stack(matrices)

        tps = matrices[:, 1, 1]
        fns = matrices[:, 1, 0]
        fps = matrices[:, 0, 1]

        recalls = tps / (tps + fns)
        precisions = tps / (tps + fps)

        rec = np.nanmean(recalls)
        prec = np.nanmean(precisions)

        fscores = 2 * precisions * recalls / (precisions + recalls)
        fsc = np.nanmean(fscores)

        auc_scores = np.array(aucs)
        auc = np.mean(auc_scores)

        aupr_scores = np.array(aupr)
        aupr = np.mean(aupr_scores)

        mcorr_scores = np.array(mcorrs)
        mcorr = np.mean(mcorr_scores)

        efective_labels = None
        efective_probs = None

    if epoch == 'test':
        efective_labels = [lbl[:l] for lbl, l in zip(labels, lengths)]
        efective_probs = [p[:l] for p, l in zip(probs, lengths)]

        # plot_roc_curve(efective_labels, efective_probs)
        # plot_pr_curve(labels, probs).show()
        f = open("results/{}/{}/cv_{}.txt".format(config['input_type'], file_name, cv), "w")
        f.write(f"Recall = {rec:.3f}\n")
        f.write(f"Precision = {prec:.3f}\n")
        f.write(f"F-score = {fsc:.3f}\n")
        f.write(f"ROC-AUC = {auc:.3f}\n")
        f.write(f"pr-AUC = {aupr:.3f}\n")
        f.write(f"MCC = {mcorr:.3f}\n")
        f.close()

        print(f"Epoch   {epoch}, Recall    {rec:.3f}, Precision  {prec:.3f}, F1-Score   {fsc:.3f}, Roc  {auc:.3f}, pr   {aupr:.3f}, mcc {mcorr:.3f}")

    return {
        'method_name': file_name + "_" + str(cv),
        'Youden': threshold,
        'Recall': rec,
        'Precision': prec,
        'F-score': fsc,
        'ROC': auc,
        'PR': aupr,
        'MCC': mcorr,
        'eff_labels': efective_labels,
        'eff_probs': efective_probs,
    }
