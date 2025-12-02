import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Function
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

nbins=5

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
            torch.nn.Linear(32, nbins)  # regression on psum/pz
        )

    def forward(self, x, lambda_):
        x = grad_reverse(x, lambda_)
        return self.net(x)

# Only load specific branches

branches=[]
branches.extend(["vertex.invM_","vertex.pos_","vertex.p1_","vertex.p2_"])

cut_branch = "vertex.invM_"
cut_threshold = 3.0
files_and_labels = [
    ("merged_tritrig_pulser_recon_tomstyle.root",0),
    ("merged_simp_pulser_60_recon_tomstyle.root",1)
]
all_data = []
all_labels = []
cut_labels = []
flattened_names = []  # to be populated once from first file

for filename, label in files_and_labels:
    with uproot.open(f"{filename}:preselection") as tree:
        #print(tree.keys())
        arrays = tree.arrays(branches)
        
        invM = ak.to_numpy(arrays["vertex.invM_"])
        ARR1 = ak.to_numpy(arrays["vertex.pos_"])
        #for field in ARR1.dtype.names:
        #    print(field)
        z = ARR1["fZ"]
        psum_values = ak.to_numpy(arrays[cut_branch])
        cut_mask = (invM>-100) & (invM<.18) & (z>-100.0) #(invM > .001*mass-.005) & (invM < .001*mass+.005)
        if np.sum(cut_mask) == 0:
            continue  # skip empty batch

        data_parts = []
        feature_names = []

        for b in branches:
            if ((b=="vertex.pos_")or(b=="vertex.invM_")):
                continue
            arr = ak.to_numpy(arrays[b])[cut_mask]
            # Handle structured dtypes (e.g., vertex.pos_)
            if arr.dtype.fields is not None:
                for field in arr.dtype.names:
                    if ((field=="fX")or(field=="fZ")):
                        continue
                    subarr = arr[field]
                    if not np.issubdtype(subarr.dtype, np.number):
                        print(f"Skipping {b}.{field}: non-numeric dtype {subarr.dtype}")
                        continue
                    data_parts.append(subarr.reshape(-1, 1))
                    feature_names.append(f"{b}.{field}")
            elif np.issubdtype(arr.dtype, np.number):
                if arr.ndim == 1:
                    data_parts.append(arr.reshape(-1, 1))
                    feature_names.append(b)
                elif arr.ndim == 2:
                    for i in range(arr.shape[1]):
                        data_parts.append(arr[:, i].reshape(-1, 1))
                        feature_names.append(f"{b}[{i}]")
                else:
                    print(f"Skipping {b}: unsupported shape {arr.shape}")
            else:
                print(f"Skipping {b}: non-numeric dtype {arr.dtype}")
        #for i, part in enumerate(data_parts):
            #print(f"Array {i} shape: {part.shape}")

        data = np.hstack(data_parts)
        labels = np.full(data.shape[0], label)

        # Example: 5 bins (adjust as needed)
        n_bins = 5
        print(psum_values[cut_mask].min())
        print(psum_values[cut_mask].max())
        bins = np.linspace(psum_values[cut_mask].min(), psum_values[cut_mask].max(), n_bins + 1)
        cut_label = np.digitize(psum_values[cut_mask], bins=bins) - 1
        cut_label = np.clip(cut_label, 0, n_bins - 1)

        all_data.append(data)
        all_labels.append(labels)
        cut_labels.append(cut_label)

        if not flattened_names:
            flattened_names = feature_names

# Combine everything
X = np.concatenate(all_data)
Y = np.concatenate(all_labels)
Z = np.concatenate(cut_labels)

# Create a composite label for stratification; this line will keep the relative ratios of signal and background as well as control and not control constant.

#composite_labels = Y.astype(str) + "_" + Z.astype(str)

composite_labels = np.array([f"{y}_{z}" for y, z in zip(Y, Z)])

X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
    X, Y, Z, test_size=0.3, random_state=42, stratify=Y
)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train, dtype=torch.long),
                              torch.tensor(Z_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(Y_test, dtype=torch.long),
                             torch.tensor(Z_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

classifier = Classifier(input_dim=X.shape[1])
adversary = Adversary(input_dim=1)  # match classifier output logits

opt_clf = torch.optim.Adam(classifier.parameters(), lr=1e-3)
opt_adv = torch.optim.Adam(adversary.parameters(), lr=1e-3)

criterion_clf = torch.nn.BCEWithLogitsLoss()
criterion_adv = torch.nn.CrossEntropyLoss()
lambda_adv = 0.001

for epoch in range(200):
    for x, y, ycut in train_loader:
        # Step 1: forward through classifier
        logits = classifier(x)
    
        # Step 2: classification loss
        loss_clf = criterion_clf(logits, y.float())
    
        # Step 3: adversarial prediction (on logits or softmax)
        adv_input = logits.detach().unsqueeze(1)  # optionally softmax(logits) if more stable
        dz_logits = adversary(adv_input, lambda_=lambda_adv)
        loss_adv = criterion_adv(dz_logits, ycut)
    
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


y_true_signal = []
y_pred_signal = []

y_true_region = []
y_pred_region_logits = []

with torch.no_grad():
    for x_batch, y_signal, y_region in test_loader:
        logits_signal = classifier(x_batch)
        probs_signal = torch.sigmoid(logits_signal)
        logits_region = adversary(logits_signal.unsqueeze(1), lambda_=0.0)

        y_true_signal.extend(y_signal.numpy())
        y_pred_signal.extend(probs_signal.numpy())

        y_true_region.extend(y_region.numpy())
        y_pred_region_logits.extend(logits_region.numpy())  # shape: (B, n_bins)

unique, counts = np.unique(y_true_signal, return_counts=True)
print(dict(zip(unique, counts)))

y_true_region = np.array(y_true_region)
y_pred_region_logits = np.array(y_pred_region_logits)
n_bins = y_pred_region_logits.shape[1]

unique, counts = np.unique(y_true_region, return_counts=True)
print(dict(zip(unique, counts)))

# Binarize true labels for one-vs-rest
y_true_bin = label_binarize(y_true_region, classes=list(range(n_bins)))

for i in range(n_bins):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_region_logits[:, i])
    
    try:
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    except:
        print(f"Class {i} has insufficient samples for AUC")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Adversary ROC (multi-class psum)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Adversary_ROC_MulticlassTestNoMassDiscrFarZMin.png")

def plot_roc(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
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
        helper=str(title)+"NoMassDiscrFarZMin.png"
        plt.savefig(helper)
    except ValueError:
        print("One of the ROC Cuves Failes")

# Classifier ROC (signal vs background)
plot_roc(y_true_signal, y_pred_signal, "Classifier_ROC_Signal_vs_BackgroundTest")

y_true_signal = []
y_pred_signal = []

y_true_region = []
y_pred_region_logits = []

with torch.no_grad():
    for x_batch, y_signal, y_region in train_loader:
        logits_signal = classifier(x_batch)
        probs_signal = torch.sigmoid(logits_signal)
        logits_region = adversary(logits_signal.unsqueeze(1), lambda_=0.0)

        y_true_signal.extend(y_signal.numpy())
        y_pred_signal.extend(probs_signal.numpy())

        y_true_region.extend(y_region.numpy())
        y_pred_region_logits.extend(logits_region.numpy())  # shape: (B, n_bins)

y_true_region = np.array(y_true_region)
y_pred_region_logits = np.array(y_pred_region_logits)
n_bins = y_pred_region_logits.shape[1]

# Binarize true labels for one-vs-rest
y_true_bin = label_binarize(y_true_region, classes=list(range(n_bins)))

for i in range(n_bins):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_region_logits[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Adversary ROC (multi-class psum)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Adversary_ROC_MulticlassTrainNoMassDiscrFarZMin.png")

def plot_roc(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
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
        helper=str(title)+"NoMassDiscrFarZMin.png"
        plt.savefig(helper)
    except ValueError:
        print("One of the ROC values failed 2")
# Classifier ROC (signal vs background)
plot_roc(y_true_signal, y_pred_signal, "Classifier_ROC_Signal_vs_BackgroundTrain")

import torch
import matplotlib.pyplot as plt
import numpy as np

# Assuming `classifier` is your trained model and `data_loader` provides the data

logits_signal = []
logits_background = []

classifier.eval()
with torch.no_grad():
    for x_batch, y_batch, y_region in test_loader:
        logits = classifier(x_batch).squeeze()  # This should be the *pre-sigmoid* value
        for logit, label in zip(logits, y_batch):
            if label == 1:  # Signal
                logits_signal.append(logit.item())
            else:          # Background
                logits_background.append(logit.item())

# Convert to numpy for plotting
logits_signal = np.array(logits_signal)
logits_background = np.array(logits_background)

# Plot
plt.clf()
plt.hist(logits_signal, bins=50, alpha=0.5, label='Signal', density=True, histtype='step')
plt.hist(logits_background, bins=50, alpha=0.5, label='Background', density=True, histtype='step')
plt.xlabel('Classifier Output (Pre-Sigmoid Logit)')
plt.ylabel('Density')
plt.legend()
plt.title('Logit Distribution Before Sigmoid')
plt.grid(True)
plt.savefig("SigmoidBingBongNoMassDiscrFarZMin.png")


