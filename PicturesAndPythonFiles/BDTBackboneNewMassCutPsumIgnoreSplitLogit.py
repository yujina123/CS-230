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
mass=60

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
branches.extend(["psum","vertex.invM_","vertex.pos_"])
#,"pos.track_.track_time_",, "ele.track_.track_time_"
branches.extend(["ele.track_.n_hits_", "ele.track_.d0_", "ele.track_.phi0_", "ele.track_.z0_", "ele.track_.tan_lambda_", "ele.track_.px_", "ele.track_.py_", "ele.track_.pz_", "ele.track_.chi2_","ele.track_.x_at_ecal_","ele.track_.y_at_ecal_","ele.track_.z_at_ecal_"])
branches.extend(["pos.track_.n_hits_", "pos.track_.d0_", "pos.track_.phi0_", "pos.track_.z0_", "pos.track_.tan_lambda_", "pos.track_.px_", "pos.track_.py_", "pos.track_.pz_", "pos.track_.chi2_","pos.track_.x_at_ecal_","pos.track_.y_at_ecal_","pos.track_.z_at_ecal_" ])
branches.extend(["vertex.chi2_","vertex.invMerr_"])
branches.extend(["vtx_proj_sig","vtx_proj_x_sig","vtx_proj_y_sig"])
branches.extend(["ele.track_.track_residuals_[14]","pos.track_.track_residuals_[14]"])
branches.extend(["ele.track_.lambda_kinks_[14]","pos.track_.lambda_kinks_[14]"])
branches.extend(["ele.track_.phi_kinks_[14]","pos.track_.phi_kinks_[14]"])

cut_branch = "psum"
cut_threshold = 3.0
files_and_labels = [
    ("merged_tritrig_pulser_recon_tomstyle.root",0),
    ("merged_simp_pulser_"+str(mass)+"_recon_tomstyle.root",1)
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
        psum_values = ak.to_numpy(arrays[cut_branch])
        print("merged_simp_pulser_"+str(mass)+"_recon_tomstyle.root")
        print(.001*mass-.005)
        print(.001*mass+.005)
        cut_mask = (invM > .001*mass-.005) & (invM < .001*mass+.005)
        
        if np.sum(cut_mask) == 0:
            continue  # skip empty batch
        data_parts = []
        feature_names = []

        for b in branches:
            arr = ak.to_numpy(arrays[b])[cut_mask]
            # Handle structured dtypes (e.g., vertex.pos_)
            if arr.dtype.fields is not None:
                for field in arr.dtype.names:
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
       
        data = np.hstack(data_parts)
        labels = np.full(data.shape[0], label)
        #cut_label = (psum_values[cut_mask] >= cut_threshold).astype(int)

        # Example: 5 bins (adjust as needed)
        n_bins = 5
        bins = np.linspace(psum_values.min(), psum_values.max(), n_bins + 1)
        cut_label = np.digitize(psum_values[cut_mask], bins=bins) - 1
        cut_label = np.clip(cut_label, 0, n_bins - 1)

        #cut_label = np.digitize(psum_values[cut_mask], bins=np.linspace(psum_values.min(), psum_values.max(), n_bins + 1), right=True) - 1
        #cut_label = np.digitize(psum_values[cut_mask], bins=np.linspace(psum_values.min(), psum_values.max(), n_bins+1)) - 1
        #cut_label = (psum_values[cut_mask] >= cut_threshold).astype(int)
        #data = np.column_stack([ak.to_numpy(arrays[b])[cut_mask] for b in branches])
        #labels = np.full(np.sum(cut_mask), label)

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
    X, Y, Z, test_size=0.3, random_state=42, stratify=composite_labels
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
lambda_adv = 0.1

for epoch in range(10):
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
plt.savefig("Adversary_ROC_MulticlassTest.png")

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
plt.savefig("Adversary_ROC_MulticlassTrain.png")

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
plt.savefig("SigmoidBingBong.png")


