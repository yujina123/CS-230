import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Function
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)  # 1-class output
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class Adversary(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)  # regression on psum/pz
        )

    def forward(self, x, lambda_):
        x = grad_reverse(x, lambda_)
        return self.net(x).squeeze(-1)

def rank_features(X, Y, Z, branches, n_epochs=3):
    base_auc_clf, base_auc_adv = evaluate_model(X, Y, Z)

    importances = []
    for i, name in enumerate(branches):
        print(f"Testing without feature: {name}")
        X_reduced = np.delete(X, i, axis=1)
        auc_clf, auc_adv = evaluate_model(X_reduced, Y, Z, n_epochs=n_epochs)
        drop_score = base_auc_clf - auc_clf
        leak_score = auc_adv - base_auc_adv
        importances.append((name, drop_score, leak_score))

    return sorted(importances, key=lambda x: (x[1] + x[2]), reverse=True)

def evaluate_model(X, Y, Z, n_epochs=3):
    # Train your classifier+adversary for a few epochs
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(Y, dtype=torch.long),
                            torch.tensor(Z, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    clf = Classifier(X.shape[1])
    adv = Adversary(1)

    opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)
    opt_adv = torch.optim.Adam(adv.parameters(), lr=1e-3)

    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        for x, y, z in loader:
            logits = clf(x)
            loss_clf = criterion(logits, y.float())
            adv_input = logits.detach().unsqueeze(1)
            loss_adv = criterion(adv(adv_input, lambda_=0.1), z.float())

            total_loss = loss_clf + 0.1 * loss_adv
            opt_clf.zero_grad()
            total_loss.backward(retain_graph=True)
            opt_clf.step()

            opt_adv.zero_grad()
            loss_adv.backward()
            opt_adv.step()

    # Evaluate
    y_true_signal, y_pred_signal, y_true_region, y_pred_region = [], [], [], []
    with torch.no_grad():
        for x, y, z in loader:
            logits = clf(x)
            probs_signal = torch.sigmoid(logits)
            probs_region = torch.sigmoid(adv(logits.unsqueeze(1), lambda_=0.0))

            y_true_signal.extend(y.numpy())
            y_pred_signal.extend(probs_signal.numpy())
            y_true_region.extend(z.numpy())
            y_pred_region.extend(probs_region.numpy())

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true_signal, y_pred_signal), roc_auc_score(y_true_region, y_pred_region)

def prune_features(X, Y, Z, ranking, names, auc_drop_thresh=0.0, adv_gain_thresh=0.05):
    keep = [i for i, (_, d_clf, d_adv) in enumerate(ranking)
            if d_clf > auc_drop_thresh or d_adv < adv_gain_thresh]
    X_pruned = X[:, keep]
    pruned_names = [names[i] for i in keep]
    return X_pruned, pruned_names

def plot_roc(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    helper=str(title)+".png"
    plt.savefig(helper)

# Only load specific branches
branches = ["ele.track_.n_hits_", "ele.track_.d0_", "ele.track_.phi0_", "ele.track_.z0_", "ele.track_.tan_lambda_", "ele.track_.px_", "ele.track_.py_", "ele.track_.pz_", "ele.track_.chi2_"]
branches.extend(["pos.track_.n_hits_", "pos.track_.d0_", "pos.track_.phi0_", "pos.track_.z0_", "pos.track_.tan_lambda_", "pos.track_.px_", "pos.track_.py_", "pos.track_.pz_", "pos.track_.chi2_", "psum"])

branches.extend(["ele.track_.track_residuals_[14]","pos.track_.track_residuals_[14]"])
branches.extend(["ele.track_.lambda_kinks_[14]","pos.track_.lambda_kinks_[14]"])
branches.extend(["ele.track_.phi_kinks_[14]","pos.track_.phi_kinks_[14]"])

cut_branch = "psum"
cut_threshold = 3.0
files_and_labels = [
    ("merged_tritrig_pulser_recon_tomstyle.root",0),
    ("merged_simp_pulser_60_recon_tomstyle.root",1)
]
all_data = []
all_labels = []
cut_labels = []

for filename, label in files_and_labels:
    with uproot.open(f"{filename}:preselection") as tree:
        arrays = tree.arrays(branches)
        #for b in branches:
        #    print(b, ak.to_numpy(arrays[b]).shape)
        psum_values = ak.to_numpy(arrays[cut_branch])
        cut_label = (psum_values>=cut_threshold).astype(int)

        # Stack branches column-wise
        data = np.column_stack([ak.to_numpy(arrays[b]) for b in branches])
        labels = np.full(len(arrays["psum"]), label)

        all_data.append(data)
        all_labels.append(labels)
        cut_labels.append(cut_label)
# Combine everything
X = np.concatenate(all_data)
Y = np.concatenate(all_labels)
Z = np.concatenate(cut_labels)
#np.concatenate(all_labels)

# Optional: wrap in PyTorch DataLoader
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                        torch.tensor(Y, dtype=torch.long),
                        torch.tensor(Z,dtype=torch.long))
loader = DataLoader(dataset, batch_size=128, shuffle=True)

classifier = Classifier(input_dim=X.shape[1])
adversary = Adversary(input_dim=1)  # match classifier output logits

opt_clf = torch.optim.Adam(classifier.parameters(), lr=1e-3)
opt_adv = torch.optim.Adam(adversary.parameters(), lr=1e-3)

criterion_clf = torch.nn.BCEWithLogitsLoss()
criterion_adv = torch.nn.BCEWithLogitsLoss()
lambda_adv = 0.1

for x, y, ycut in loader:
    # Step 1: forward through classifier
    logits = classifier(x)

    # Step 2: classification loss
    loss_clf = criterion_clf(logits, y.float())

    # Step 3: adversarial prediction (on logits or softmax)
    adv_input = logits.detach().unsqueeze(1)  # optionally softmax(logits) if more stable
    dz_logits = adversary(adv_input, lambda_=lambda_adv)
    loss_adv = criterion_adv(dz_logits, ycut.float())

    # Step 5: combined loss (gradient flows through classifier only from clf_loss)
    total_loss = loss_clf + lambda_adv * loss_adv

    # Step 6: update classifier
    opt_clf.zero_grad()
    total_loss.backward(retain_graph=True)  # keep graph for adversary update
    opt_clf.step()

    # Step 7: update adversary separately
    opt_adv.zero_grad()
    loss_adv.backward()
    opt_adv.step()

flattened_data = []
flattened_names = []

for b in branches:
    arr = ak.to_numpy(arrays[b])
    if arr.ndim == 2:
        # Expand multi-element arrays like track_residuals_[14]
        for i in range(arr.shape[1]):
            flattened_data.append(arr[:, i])
            flattened_names.append(f"{b}[{i}]")
    else:
        flattened_data.append(arr)
        flattened_names.append(b)

data = np.column_stack(flattened_data)

ranking = rank_features(X, Y, Z, flattened_names, n_epochs=3)
for name, drop_score, leak_score in ranking:
    print(f"{name:50s} | ΔAUC_clf: {drop_score:.4f}, ΔAUC_adv: {leak_score:.4f}")

features_to_remove = [name for name, d_clf, d_adv in ranking
                      if d_clf > 0 and d_adv < 0.05]

X_pruned, names_pruned = prune_features(X, Y, Z, ranking, flattened_names)
auc_clf, auc_adv = evaluate_model(X_pruned, Y, Z, n_epochs=5)
print(f"Post-prune AUC: classifier={auc_clf:.3f}, adversary={auc_adv:.3f}")

ranking = rank_features(X_pruned, Y, Z, flattened_names, n_epochs=3)
for name, drop_score, leak_score in ranking:
    print(f"{name:50s} | ΔAUC_clf: {drop_score:.4f}, ΔAUC_adv: {leak_score:.4f}")

y_true_signal = []
y_pred_signal = []

y_true_region = []
y_pred_region = []

with torch.no_grad():
    for x_batch, y_signal, y_region in loader:
        # Forward through classifier
        logits_signal = classifier(x_batch)
        probs_signal = torch.sigmoid(logits_signal)  # [B]

        # Adversary tries to predict region label from classifier output
        logits_region = adversary(logits_signal.detach().unsqueeze(1), lambda_=0.0)  # no grad reversal
        probs_region = torch.sigmoid(logits_region)  # [B]

        # Collect true and predicted labels
        y_true_signal.extend(y_signal.numpy())
        y_pred_signal.extend(probs_signal.numpy())

        y_true_region.extend(y_region.numpy())
        y_pred_region.extend(probs_region.numpy())

print("Unique region labels:", np.unique(y_true_region))
print("Min/max of predicted region scores:", np.min(y_pred_region), np.max(y_pred_region))



# Classifier ROC (signal vs background)
plot_roc(y_true_signal, y_pred_signal, "Classifier_ROC_Signal_vs_Background")

# Adversary ROC (control vs signal region)
plot_roc(y_true_region, y_pred_region, "Adversary_ROC_Control_vs_Signal_Region")

