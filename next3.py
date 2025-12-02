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
import argparse
from sklearn.preprocessing import label_binarize
##YB for recognizing overflow, underflow
def print_bin_stats(values, edges, tag):
    """
    values: 1D numpy array of numbers (your feature values)
    edges : 1D numpy array of bin edges (length = n_bins + 1)
    tag   : label for the printout, e.g. 'train_sig'
    """
    # per-bin counts
    counts, _ = np.histogram(values, bins=edges)
    
    # under/overflow
    under = (values <  edges[0]).sum()
    over  = (values >= edges[-1]).sum()
    total = values.size
    
    print(f"{tag}: total={total}, underflow={under}, overflow={over}")
    print(f"  per-bin counts: {counts.tolist()}")
    return counts, under, over

#########SETS THE NUMBER OF ADVERSARIAL CLASSES
nbins=2 ##YB how many categories adversary network tries to predict  #ADV-ED COMMENTED OUT CHANGED FROM 5 TO 2

##YB reads command-line flag --epochs or -e to set how many passes through the data to make during training
parser = argparse.ArgumentParser(description="describe epoch numbers plot")
parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of training epochs")
args = parser.parse_args()
EPOCH=args.epochs

##YB Gradient Reversal Layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    ##YB in backward pass multiply all gradients by -lambda, classifier pushes features that fool the adversary
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


####MAIN NEURAL NETWORK FOR CLASSIFICATION
##YB input layer: take input_dim features per event
##YB hidden layer: 64 neurons with a ReLu activation - give non linearity
##YB output layer: a logit per event -> prob of signal vs bkg

#TO DO: get classifier object to extract hidden layer info, keep its ability to more than one hidden layer, i want certain dim and extract their nodes
#TO DO: want the number associated than the matrix, output after eval number itself extract
#TO DO: not care about weights with the matrices, actual calculation the node does, weight + inputs
class Classifier(torch.nn.Module):
    #def __init__(self, input_dim, hidden_dim=64):
    #2 hidden layers: def __init__(self, input_dim, hidden_dim=64, hidden_dim2=64):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            #2 hidden layers: torch.nn.Linear(hidden_dim, hidden_dim2), #TO CHANGEEEEE
            #2 hidden layers: torch.nn.Linear(hidden_dim2, 1)  # 1-class output
            torch.nn.Linear(hidden_dim, 1)  # 1-class output
        )

    def forward(self, x):
        #ADV-ED COMMENTED OUT
        #NEW EDITION BELOW
        """
        features = self.feature_extractor(x)        # hidden layer output
        out = self.classifier_head(features).squeeze(-1)
        return out, features                         # return both
         """
        coding = self.net[1](self.net[0](x))              # hidden layer output
        hidden_pre_relu = self.net[0](x)
        if torch.isnan(hidden_pre_relu).any() or torch.isinf(hidden_pre_relu).any():
            print('[DIAG] NaNs or Infs in Linear layer output')
        hidden_post_relu = self.net[1](hidden_pre_relu)
        if torch.isnan(hidden_post_relu).any() or torch.isinf(hidden_post_relu).any():
            print('[DIAG] NaNs or Infs after ReLU')
        #print('[DIAG] Hidden pre-ReLU shape:', hidden_pre_relu.shape)
        #print('[DIAG] Hidden post-ReLU shape:', hidden_post_relu.shape)
        coding = hidden_post_relu
        out = self.net[2](coding).squeeze(-1)             # classifier output
       #TRY THIS?
        """ if torch.isnan(coding).any():
            print("⚠️ NaNs detected in coding!")
            print("Hidden values before ReLU:", self.net[0](x))
            print("Input x:", x)
        """
        return out, coding                               # return both
        #return self.net(x).squeeze(-1)

######OTHER NEURAL NETWORK FOR ADVERSARIAL TASK
##YB takes as input the classifier's internal representation - raw logit after applying gradient reversal
##YB passes it through a small network (32-unit hiddent layer) and outputs nbins scores
##YB nbins scores correspond to your 5 bins (the "adversarial" task of predicting which psum-range bin an event falls into)

## YB longer explaination: For each event, classifier produces a single number called a logit - unsquashed "confidence" score for signal vs. background.
## YB We take that logit, wrap it in a tiny vector, and feed it to adversary network. Before adversary's layers see data, 
## YB pass it thru gradient reversal operation. In the forward direction, gradient reversal is the identity function - return logit
## YB unchanged. The adversary processes that logit through 32-neuron hidden layer + ReLu and a 5-neuron output layer.
## YB The five numbers are raw score (one per bin) that the adversary will use to predict which of the five psum ranges the original event came from.

class Adversary(torch.nn.Module):
    def __init__(self, input_dim):
         #ADV-ED COMMENTED OUT
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, nbins)  # regression on psum/pz
        )

    def forward(self, x, lambda_ = 1): # #ADV-ED COMMENTED OUT CHANGED FROM JUST LAMBDA_ TO LAMBDA_ = 1
        x = grad_reverse(x, lambda_)
        return self.net(x)

## YB specifies which physics variables to pull from two ROOT files (one 0 and another 1)
## YB prepares empty lists to accumulate your features, signal/background labels, and adversary-bin labels
    
# Only load specific branches
#############THIS CONTAINS ALL THE BRANCHES OF THE ROOT FILE WE AIM TO USE; ADD MORE NAMES AS YOU SEE FIT (OR REMOVE)#############
branches=[]
branches.extend(["psum","vertex.invM_","vertex.pos_"])
#,"pos.track_.track_time_",, "ele.track_.track_time_"
branches.extend(["ele.track_.n_hits_", "ele.track_.d0_", "ele.track_.phi0_", "ele.track_.z0_", "ele.track_.tan_lambda_", "ele.track_.px_", "ele.track_.py_", "ele.track_.pz_", "ele.track_.chi2_","ele.track_.x_at_ecal_","ele.track_.y_at_ecal_","ele.track_.z_at_ecal_"])
branches.extend(["pos.track_.n_hits_", "pos.track_.d0_", "pos.track_.phi0_", "pos.track_.z0_", "pos.track_.tan_lambda_", "pos.track_.px_", "pos.track_.py_", "pos.track_.pz_", "pos.track_.chi2_","pos.track_.x_at_ecal_","pos.track_.y_at_ecal_","pos.track_.z_at_ecal_" ])
branches.extend(["vertex.chi2_","vertex.invMerr_"])
branches.extend(["vtx_proj_sig","vtx_proj_x_sig","vtx_proj_y_sig"])
#branches.extend(["ele.track_.track_residuals_[14]","pos.track_.track_residuals_[14]"])
#branches.extend(["ele.track_.lambda_kinks_[14]","pos.track_.lambda_kinks_[14]"])
#branches.extend(["ele.track_.phi_kinks_[14]","pos.track_.phi_kinks_[14]"])

###############THIS ESTABLISHES WHAT BRANCH WE ARE MAKING ADVERSARIAL########
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


## YB opens each file's preselection TTree, reads only the desired branches into memory, applies a mask,
## YB flattens each branch into a single 2D feature array, labels each event as signal (1) or background (0)
## YB divides the chosen variable (psum) range into 5 equal bins and assigns each event a bin index from 0 to 4. 
## YB accumulates all data, labels, and bin-labels

###YB for smaller mass
### when fills up to 10,000 events move on to next file
#################THIS LOADS IN ALL THE EVENTS FROM THE SIGNAL AND BACKGROUND FILE WITH A MASK IF YOU WANT DISPLACED VERTICES#############
for filename, label in files_and_labels:
    #YB TO SKIP 10K EVENTS event_count = 0
    with uproot.open(f"{filename}:preselection") as tree:
        #print(tree.keys())
        arrays = tree.arrays(branches)
        invM = ak.to_numpy(arrays["vertex.invM_"])
        ARR1 = ak.to_numpy(arrays["vertex.pos_"])
        #for field in ARR1.dtype.names:
        #    print(field)
        z = ARR1["fZ"]
        psum_values = ak.to_numpy(arrays[cut_branch])
        #############THIS LINE DOES A CUT ON DISPLACE VERTICES LENGTH (Z) AS WELL AS INVARIANT MASS (THOUGH THAT CUT IS SO LOOSE AS TO BE VOID)############
        cut_mask = (invM>-100) & (invM<.18)# & (z>10.0) #(invM > .001*mass-.005) & (invM < .001*mass+.005)
        if np.sum(cut_mask) == 0:
            continue  # skip empty batch

        data_parts = []
        feature_names = []
        #############DIFFERENT BRANCHES HAD WIDELY VARYING STRUCTURE, THIS LOOP IS MEANT TO STORE EACH AS A 1D VARIABLE##########
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
        #for i, part in enumerate(data_parts):
            #print(f"Array {i} shape: {part.shape}")

        data = np.hstack(data_parts)
        """n_new = data.shape[0] # NEW ADDED YB!!! to quit after 10k events count how many events this chunk adds
        print(n_new)
         # if adding these would exceed 10k, stop and move on
        if event_count + n_new > MAX_EVENTS_PER_FILE:
            print(f"Reached {event_count} + {n_new} > {MAX_EVENTS_PER_FILE} in {filename}, skipping the rest")
            break
        
        # otherwise accept them
        event_count += n_new
        """
        labels = np.full(data.shape[0], label)

        ############THIS STORES THE DATA INTO all_data ALONGSIDE labels TO TELL YOU SIGNAL OR BACKGROUND AND cut_labels FOR WHAT ADVERSARY REGION YOU'RE IN
        # Example: 5 bins (adjust as needed)
        n_bins = nbins
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

###########THIS IS THE MEAT OF THE CODE, THE PORTION DOING TRAINING. FIRST WE MUST STAGE THE DATA A BIT#############
            
# Combine everything
## YB concatenates all events into single feature matrix X, signal label vector Y, and bin label vector Z.
## YB splits into training (70%) and testing (30%), making sure the fraction of signal vs. background stays the same in both sets.
## YB wraps them in PyTorch DataLoaders so you can iterate in minibatches of 128 events.
X = np.concatenate(all_data)
Y = np.concatenate(all_labels)
Z = np.concatenate(cut_labels)
    
# Create a composite label for stratification; this line will keep the relative ratios of signal and background as well as control and not control constant.

#composite_labels = Y.astype(str) + "_" + Z.astype(str)

composite_labels = np.array([f"{y}_{z}" for y, z in zip(Y, Z)])
####################THIS ESTABLISHED THE TRAIN AND TESTING SET, BY SETTING ASIDE RANDOM EVENTS AT A RATE OF 1/3 FOR TRAINING, IT KEEPS Y (THE AMOUNT OF SIGNAL) PROPORTIONAL IN EACH SET###############
X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
    X, Y, Z, test_size=0.3, random_state=42, stratify=Y
)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train, dtype=torch.long),
                              torch.tensor(Z_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(Y_test, dtype=torch.long),
                             torch.tensor(Z_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=128, drop_last=True)

##########WE NOW ESTABLISH THE CLASSIFIER AND ADVERSARY NETWORKS
## YB creates the two networks, sets up Adam optimizers for both. 
## YB Chooses loss functions: Binary cross-entropy with logits for the classifier (signal vs background)
## YB                     and Cross-entropy for the adversary (multi-class psum bin classification), 
## YB assigns a mixing weight = 50 to control how strongly the adversary affects training. 
classifier = Classifier(input_dim=X.shape[1])
##YB TO DO: Change the number of dimensions
#changed from 1 to 5
adversary = Adversary(input_dim=64)  # match classifier output logits

###########THESE ARE THEIR PARAMETERS
opt_clf = torch.optim.Adam(classifier.parameters(), lr=1e-3)
opt_adv = torch.optim.Adam(adversary.parameters(), lr=1e-3)

#########THIS IS THE LOSS FUNCTION FOR EACH
# Extracting the hidden node information to classifier to feed into the adversary 
criterion_clf = torch.nn.BCEWithLogitsLoss()
criterion_adv = torch.nn.CrossEntropyLoss()
lambda_adv = 1 #set this to 1 (BETTER ROC curves but leakages, if high better leakage cover but worse performance - to make both of them good, we add more layers of classifier as inputs to the adversary) INITIALLY WAS 50 
#leakage: force neural network to invariant with respect to whatever the adversary is killing: rn mass, more leakage = classifier using mass to classify/predict
    
##YB ROC CURVE FOR EPOCH 0 - EPOCH 1
roc_snaps = {}

##YB helper functions to get true labels and scores on the test dataset
def get_test_scores():
    """Run the classifier on the test set and return:
    - y_true array: the actual labels (0 or 1 - signal or background)
    - y_score_chunks array: the predicted probability of '1' for each event"""

    #Put the model in evaluation mode so it doesn't change its internal state
    classifier.eval()

    #Collect results in lists, then combine into big arrays
    y_true_chunks = []
    y_score_chunks = []

    #"no_grad" means to not track gradients as this is for testing
    with torch.no_grad():
        #loop over all batches in the test_loader
        for xb, yb, _ in test_loader:
            #xb: batch of input features (shape: batch_size x num_features)
            #yb": true labels for that batch
            #"_": unused adversary label
            
            # 1) run the classifer: get raw outputs called "logits"
            logits, coding = classifier(xb)
            # 2) convert logits to probabilities via the sigmoid function
            probs  = torch.sigmoid(logits)
            # 3) move data to NumPy and store
            y_true_chunks.append(yb.numpy())
            y_score_chunks.append(probs.numpy())
    #return model to "training mode" so subsequent calls modify weights again
    classifier.train()

    #concatenate all chuncks to single arrays and return
    y_true  = np.concatenate(y_true_chunks)
    y_score = np.concatenate(y_score_chunks)
    return y_true, y_score
##YB helper function to take a snapshot
def take_roc_snapshot(stage_name):
    """Compute test-set scores and store them in the global
       roc_snaps dict under the given key"""
    y_true, y_score = get_test_scores()
    roc_snaps[stage_name] = (y_true, y_score)

##########WE NOW TRAIN OVER THE SAME TRAINING SET epoch MANY TIMES TO GET STABLE PERFORMANCE
## YB forward pass through classifier to get a logit per event, compute classification loss 
## YB comparing to the true signal/background label, freeze the classifier (using .detach()) and pass
## YB its output through the adversary - applying gradient reversal internally, compute adversary loss on the
## YB true bin labels, train the classifier to both minimize its own loss and (because of gradient reversal)
## YB make the adversary's job harder, train the adversary to get better at its bin-classification task.


##YB prepare lists to record loss values over epochs  
train_loss_list = [] #total training loss per epoch
avgTrain_loss_list = [] #average training loss per epoch
avgValidate_loss_list = [] #average validation loss per epoch

## YB EPOCH 0 baseline (no training/no weight updates)
classifier.eval() #evaluation mode - no weight changes

# training data baseline loss
pre_train_sum = 0.0; pre_train_batches = 0
with torch.no_grad():
    for xb, yb, _ in train_loader:
        logits, coding = classifier(xb)
        # criterian_clf is a function measuring "binary cross-entropy"
        # it returns a small tensor representing how far off the prediction is
        loss   = criterion_clf(logits, yb.float())
        pre_train_sum     += loss.item() #accumulate each batch
        pre_train_batches += 1 #count batches
pre_train_avg = pre_train_sum / pre_train_batches

# validation data baseline loss
pre_val_sum = 0.0; pre_val_batches = 0
with torch.no_grad():
    for xb, yb, _ in test_loader:
        logits, coding = classifier(xb)
        loss   = criterion_clf(logits, yb.float())
        pre_val_sum     += loss.item()
        pre_val_batches += 1
pre_val_avg = pre_val_sum / pre_val_batches

#DEBUGGG REMOVE LATER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)
adversary.to(device)

# resume training mode so weights updates
classifier.train()  
#to debug
adversary.train()
##YB FOR VALIDATION LOSS: 
# (2) Collect per‑batch losses at epoch 0  ← ADDED
train_batch_losses0 = []
with torch.no_grad():
    for xb, yb, _ in train_loader:
        logits, coding = classifier(xb)
        train_batch_losses0.append(
            criterion_clf(logits, yb.float()).item()
        )

val_batch_losses0 = []
with torch.no_grad():
    for xb, yb, _ in test_loader:
        logits, coding = classifier(xb)
        val_batch_losses0.append(
            #ADV-ED COMMENTED OUT
#      criterion_clf(classifier(xb), yb.float()).item()
            criterion_clf(logits, yb.float()).item()    
        )

# (3) Plot epoch 0 per‑batch train & val losses  ← ADDED
plt.figure()
plt.plot(train_batch_losses0, label='Train (epoch 0)')
plt.plot(val_batch_losses0,   label='Val   (epoch 0)')
plt.xlabel("Batch index")
plt.ylabel("Loss")
plt.title("Epoch 0: Batch training & validation loss")
plt.legend()
plt.tight_layout()
plt.savefig("epoch0_train_val_loss_per_batch.png")
plt.close()

# store epoch 0 losses
train_loss_list.append(pre_train_sum)
avgTrain_loss_list.append(pre_train_avg)
avgValidate_loss_list.append(pre_val_avg)

# take ROC snapshot at epoch 0 (before any training)
take_roc_snapshot('epoch0_pretrain')
print("Total events :", X.shape[0])

MAX_EVENTS_PER_FILE = 10000
for epoch in range(1, EPOCH+1):
    #YB --------TRAINING & VALIDATION PASS ----------#
    epoch_loss_sum = num_batches = 0
    validate_loss_sum = num_val_batches = 0 
    validate_loss1_sum = num_val1_batches = 0  
    validate_loss2_sum = num_val2_batches = 0 
    #YB NEWW FOR 10K EVENTS
    samples_seen = 0
    ##YB grecord individual batch losses on epoch 1
    if epoch == 1:
        train_batch_losses1 = []
        val_batch_losses1 = []                                               # <<< ADDED
    if epoch == 2:
        train_batch_losses2 = []
        val_batch_losses2 = []                                               # <<< ADDED
    #num_train_batches_total = len(train_loader)
    #halfway_batch_index = num_train_batches_total // 2


    ##YB MY CODE
    #make a counter until 10 k and exit
    for x, y, ycut in train_loader:  #have to do enumerate for mid roc skipped
        #TO DEBUG THE SIZE MISMATCH
        #print(f"[DEBUG] Batch x shape: {x.shape}")
        #for i in range(x.shape[0]):
        #    print(f"  Row {i} shape: {x[i].shape}")
        
        x, y, ycut = x.to(device), y.to(device), ycut.to(device)
            # Optional: Check for NaNs/Infs

        #YB!!! FOR 10K EVENT @@@
        batch_size = x.size(0)
        # if taking this batch would exceed 10k, stop training for this epoch
        if samples_seen + batch_size > MAX_EVENTS_PER_FILE:
            print(f"Epoch {epoch}: reached {samples_seen} samples, stopping at 10k")
            break

        # Step 1: forward through classifier
         #ADV-ED COMMENTED OUT FOMR LOGITS = CLASSIFIER(X) TO:
        logits, coding = classifier(x)

        # Diagnostics for mismatch
        if coding.shape[0] != x.shape[0]:
            print("⚠️ Skipping batch: mismatch between input and coding shape")
            print(f"x.shape = {x.shape}")
            print(f"coding.shape = {coding.shape}")
            print("Input x (sample):", x[:5])  # print only first few for brevity
            print("Coding (sample):", coding[:5])
            # assert coding.shape[0] == x.shape[0], "Mismatch in batch size between x and coding"
            continue
        #CHECK OF NaN for batch size?
        """if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print("Input x contains NaNs or Infs! Skipping this batch.")
            continue
        """
        if coding.shape[0] != x.shape[0]:
            print("⚠️ Skipping batch: mismatch between input and output")
            continue
        # Step 2: classification loss
        loss_clf = criterion_clf(logits, y.float())

        # YB MY CODEEE: recod if in first epoch
        if epoch == 1:
            train_batch_losses1.append(loss_clf.item())
        #Test loader for loop happens, maybe you sum
            classifier.eval() 
            with torch.no_grad(): 
                for xb, yb, _ in test_loader:  
                    logits, coding = classifier(xb)                                   # <<< ADDED
                    loss1 = criterion_clf(logits, yb.float())             # <<< ADDED
                    validate_loss1_sum += loss1.item()                             # <<< ADDED THIS validate_loss_sum if for when epoch is 1
                    num_val1_batches  += 1                                        # <<< ADDED
            val_batch_losses1.append(validate_loss1_sum/num_val1_batches)  #validate_loss1_sum is each batch and so graphing it per batch
            classifier.train()   

        if epoch == 2:
            train_batch_losses2.append(loss_clf.item())
        #Test loader for loop happens, maybe you sum
            classifier.eval() 
            with torch.no_grad(): 
                for xb, yb, _ in test_loader:          
                    logits, coding = classifier(xb)                              # <<< ADDED
                    loss2 = criterion_clf(logits, yb.float())             # <<< ADDED
                    validate_loss2_sum += loss2.item()                             # <<< ADDED THIS validate_loss_sum if for when epoch is 1
                    num_val2_batches  += 1                                        # <<< ADDED
            val_batch_losses2.append(validate_loss2_sum/num_val2_batches)  #validate_loss1_sum is each batch and so graphing it per batch
            classifier.train() 
        #val_batch_losses1.append(averageovertestloader)
        ##accumulate for epoch summary
        epoch_loss_sum += loss_clf.item()
        num_batches += 1
        samples_seen += batch_size
        ## YB MY CODEEE
    
        # Step 3: adversarial prediction (on logits or softmax)
        adv_input = coding.detach() #ADV-ED COMMENTED OUT adv_input = logits.detach().unsqueeze(1)  # optionally softmax(logits) if more stable
        dz_logits = adversary(adv_input, lambda_=lambda_adv) #feeds that single node to adversary - classifier output
        #^^change to all the nodes plus the current node to classifier output
        #DEBUGGING 
        #print("dz_logits.shape:", dz_logits.shape)
        #print("ycut.shape:", ycut.shape)

#DEBUGGING
        #print(f"x.shape = {x.shape}")
        #print(f"y.shape = {y.shape}")
        #print(f"ycut.shape = {ycut.shape}")
        #print(f"coding.shape = {coding.shape}")  # this is input to adversary
        #print(f"dz_logits.shape = {dz_logits.shape}")
        # CrossEntropyLoss expects (N, C) input and (N,) targets
        #assert dz_logits.shape[0] == ycut.shape[0], "Batch size mismatch between adversary output and ycut"

        #ycut = torch.argmax(ycut, dim=1)  # Ensure proper shape for CrossEntropyLoss
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
        #TO KNOW HO WMANY EVENTS PROCESSED
        #print(f"Epoch {epoch}: processed {samples_seen}/{MAX_EVENTS_PER_FILE} training samples")
        # after fist epoch, snapshot ROC again
        if epoch == 1:
            take_roc_snapshot('epoch1_end')

    ##YB MY CODEEE: record epoch-level training loss
    avgTrain_loss_list.append(epoch_loss_sum/num_batches)
    train_loss_list.append(epoch_loss_sum)

        # — validation pass & per-batch val losses for epoch 1 —  <<< ADDED
   # validate_loss_sum = num_val_batches = 0                               # <<< ADDED
   # classifier.eval()                                                     # <<< ADDED
    with torch.no_grad():                                                 # <<< ADDED
        for xb, yb, _ in test_loader:   
            logits, coding = classifier(xb)                                     # <<< ADDED
            val_loss = criterion_clf(logits, yb.float())             # <<< ADDED
            validate_loss_sum += val_loss.item()                             # <<< ADDED
            num_val_batches  += 1                                        # <<< ADDED
            #if epoch == 1:
               # val_batch_losses1.append(loss1.item())
    classifier.train()                                                    # <<< ADDED
    avgValidate_loss_list.append(validate_loss_sum/num_val_batches)       # <<< ADDED  it was added where "was added" written            
#"was added"
    # — plot epoch 1 per batch train & val losses —  <<< ADDED
plt.figure()                                                         # <<< ADDED
plt.plot(train_batch_losses1, label='Train (epoch 1)')               # <<< ADDED
plt.plot(val_batch_losses1,   label='Val   (epoch 1)')               # <<< ADDED
plt.xlabel("Batch index")                                            # <<< ADDED
plt.ylabel("Loss")                                                   # <<< ADDED
plt.title("Epoch 1: Batch training & validation loss")               # <<< ADDED
plt.legend()                                                         # <<< ADDED
plt.tight_layout()                                                  # <<< ADDED
plt.savefig("epoch1_train_val_loss_per_batch.png")                    # <<< ADDED
plt.close()   
     
    # — plot epoch 2 per batch train & val losses —  <<< ADDED
plt.figure()                                                         # <<< ADDED
plt.plot(train_batch_losses2, label='Train (epoch 2)')               # <<< ADDED
plt.plot(val_batch_losses2,   label='Val   (epoch 2)')               # <<< ADDED
plt.xlabel("Batch index")                                            # <<< ADDED
plt.ylabel("Loss")                                                   # <<< ADDED
plt.title("Epoch 2: Batch training & validation loss")               # <<< ADDED
plt.legend()                                                         # <<< ADDED
plt.tight_layout()                                                   # <<< ADDED
plt.savefig("epoch2_train_val_loss_per_batch.png")                    # <<< ADDED
plt.close()   
"""????
"""
    
plt.figure()
plt.plot(range(0, EPOCH+1), train_loss_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("loss_clf Sum")
plt.title("Training loss/Epoch")
plt.savefig("SUM_EPOCH_LOSS PER EPOCH.png")


plt.figure()
plt.plot(range(0, EPOCH+1), avgTrain_loss_list, marker='o', label='Average Training loss')
plt.plot(range(0, EPOCH+1), avgValidate_loss_list, marker='s', label='Average Validation loss')
plt.xlabel("Epoch")
plt.ylabel("Average loss")
plt.title("Average Training and Validation loss/Epoch")
plt.legend()
plt.savefig("AVG_LOSS_PER_EPOCH_BOTHLATEST.png")

##YB MY CODEEEEE
plt.figure()
for stage, label_text in [('epoch0_pretrain','Epoch 0 (pre-train)'),
                          ('epoch1_end','Epoch 1 (end)')]:
    if stage in roc_snaps:
        y_t, y_s = roc_snaps[stage]
        fpr, tpr, _ = roc_curve(y_t, y_s)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label_text} AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--',label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Progress: Epoch 0 → Epoch 1')
plt.legend()
plt.savefig("ROC_epoch0_epoch1_progress.png")
plt.close()

#############IT IS AT THIS POINT THE ENTIRE CODE IS DIRECTED TOWARDS PLOTTING####################
#########THIS SECTION OF CODE AIMS TO GET THE CLASSIFIER PERFORMANCE ON THE TEST DATA, IN TESTLOADER##########
## YB runs the test set through the classifier, converts logits to probabilities with a sigmoid, 
## YB builds and saves ROC curve (receiver-operating characteristic), AUC (area under ROC), significance curve (signal efficiency divided by 
## YB sqrt of background efficiency)
y_true_signal = []
y_pred_signal = []

y_true_region = []
y_pred_region_logits = []

with torch.no_grad():
    for x_batch, y_signal, y_region in test_loader:
        logits_signal, coding = classifier(x_batch) # ADV-ED COMMENTED OUT - ADDED FEATURE
        probs_signal = torch.sigmoid(logits_signal)
        logits_region = adversary(coding.detach(), lambda_=1.0)

        y_true_signal.extend(y_signal.numpy())
        y_pred_signal.extend(probs_signal.numpy())

        y_true_region.extend(y_region.numpy())
        y_pred_region_logits.extend(logits_region.numpy())  # shape: (B, n_bins)

#####This section of code plots the adversarial ROC curves; I will refer to it as adversarry#############
unique, counts = np.unique(y_true_signal, return_counts=True)
print(dict(zip(unique, counts)))

y_true_region = np.array(y_true_region)
y_pred_region_logits = np.array(y_pred_region_logits)
n_bins = y_pred_region_logits.shape[1]

unique, counts = np.unique(y_true_region, return_counts=True)
print(dict(zip(unique, counts)))

# Binarize true labels for one-vs-rest
## YB measures how well each adversial bin classifer can tell "this event is in bin i vs. not bin i."
## YB saves a 5-line ROC plot, one for each bin
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
plt.savefig("Adversary_ROC_MulticlassTestNoMassDiscrFarZ.png")

##YB MY CODEEE
for idx, name in enumerate(flattened_names):
    print(idx, name)

#TO PRINT PER BIN EVENT COUNT, OVERFLOW AND UNDERFLOW !!!
    train_vals = (X_train[Y_train == 1, idx], #signal train
                X_train[Y_train == 0, idx]) #bkg train
    test_vals = (X_test [Y_test == 1, idx],   #signal test
               X_test[Y_test == 0, idx])    #bkg test
    # 1) collect *all* values for this feature to choose common bin edges
    all_vals = np.concatenate([train_vals[0], train_vals[1], test_vals[0], test_vals[1]])

    # 2) pick bin edges (50 bins here; change if you want)
    edges = np.linspace(all_vals.min(), all_vals.max(), 51)

    # 3) print stats
    print(f"\n==== {name} ====")
    print_bin_stats(train_vals[0], edges, "train_sig")
    print_bin_stats(train_vals[1], edges, "train_bkg")
    print_bin_stats(test_vals[0],  edges, "test_sig")
    print_bin_stats(test_vals[1],  edges, "test_bkg")

pz_col = 12

signal_mask_train = (Y_train == 1)
bkg_mask_train = (Y_train == 0)

pz_signal_train = X_train[signal_mask_train, pz_col]
pz_bkg_train = X_train[bkg_mask_train, pz_col]

plt.figure()
plt.hist(pz_signal_train,
    bins=50,
    density=True,
    histtype='step',
    label='Signal (training)')
plt.xlabel('ele.track_.pz_ (GeV/c)')
plt.ylabel('Density')
plt.hist(pz_bkg_train,
        bins=50,
        density=True,
        histtype='step',
        label='Bkg (training)')
plt.xlabel('ele.track_.pz_ (GeV/c)')
plt.ylabel('Density')
plt.savefig("YUJINAS FIRST TASK TRAIN- Signal vs Background.png")
plt.close()
#### YB TEST

signal_mask_test = (Y_test == 1)
bkg_mask_test = (Y_test == 0)

pz_signal_test = X_test[signal_mask_test, pz_col]
pz_bkg_test = X_test[bkg_mask_test, pz_col]

plt.figure()
plt.hist(pz_signal_test,
    bins=50,
    density=True,
    histtype='step',
    label='Signal (training)')
plt.xlabel('ele.track_.pz_ (GeV/c)')
plt.ylabel('Density')
plt.hist(pz_bkg_test,
        bins=50,
        density=True,
        histtype='step',
        label='Bkg (training)')
plt.xlabel('ele.track_.pz_ (GeV/c)')
plt.ylabel('Density')
plt.savefig("YUJINAS FIRST TASK TEST - Signal vs Background.png")
plt.close()

##YB MY CODEEEEE
print("Total events :", X.shape[0])
for idx, name in enumerate(flattened_names):
  train_vals = X_train[Y_train == 1, idx], X_train[Y_train == 0, idx] #mask and then idx means column for feature element (matrix composition: rows - event # (?), columns - feature/types like ele.pos, ele.pz)
  test_vals = X_test[Y_test == 1, idx], X_test[Y_test == 0, idx]
  #figure with two subplots
  fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(10,4), sharey=True)

  #train-set histograms
  ax_tr.hist(train_vals[1], bins=50, density=True, histtype='step', label ='Bkg(train)')
  ax_tr.hist(train_vals[0], bins=50, density=True, histtype='step', label ='Sig(train)')
  ax_tr.set_title(f'Train: {name}')
  ax_tr.set_xlabel(name)
  ax_tr.set_ylabel('Density')
  ax_tr.legend()

  #test-set histograms
  ax_te.hist(test_vals[1], bins=50, density=True, histtype='step', label ='Bkg(test)')
  ax_te.hist(test_vals[0], bins=50, density=True, histtype='step', label ='Sig(test)')
  ax_te.set_title(f'Test: {name}')
  ax_te.set_xlabel(name)
  ax_te.legend()

  fig.suptitle(f'Distribution of {name}', fontsize=15)
  fig.tight_layout()
  safe_name = name.replace('.', '_').replace('[', '_').replace(']', '')
  fig.savefig(f"Z(TO KEEP IT IN ONE PLACE): {safe_name}.png")
  plt.close()

## YB MY CODEEEEE ^^^^
############THIS PLOTS THE ROC CURVE################
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
        helper=str(title)+"NoMassDiscrFarZ.png"
        plt.savefig(helper)
    except ValueError:
        print("One of the ROC Cuves Failes")
############THIS PLOTS SIGNIFICANCE, OR THE NUMBER OF SIGNAL EVENTS OVER THE NUMBER OF BACKGROUND################
def plot_sig(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        def helper(a,b):
            if a>=.00001:
                return b/np.sqrt(a)
            return 0
        sig = [helper(fpr[i],tpr[i]) for i in range(len(tpr))] 
        roc_auc = max(sig)
        #I = min([i*(dig[i]==roc_auc) for i in range(len(dig))])
        #FPR=fpr[I]
        #TPR=fpr[I]
        #sig = []
        #for i in range(len(dig)):
        #    if (dig[i]>0.01*roc_auc)and(fpr[i]>.001):
        #        sig.append(dig[i])
        plt.figure()
        plt.plot(fpr, sig, lw=2, label=f'SIG = {roc_auc:.2f}')#, fpr = {FPR:.2f}, tpr =  {TPR:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('Scan Parameter')
        plt.ylabel('Significance')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        helper=str(title)+"NoMassDiscrFarZ"+str(EPOCH)+".png"
        plt.savefig(helper)
    except ValueError:
        print("One of the ROC Cuves Failes")

#####THIS LINE PLOTS THE ROC FOR THE ACTUAL CLASSIFIER CLASS#############
# Classifier ROC (signal vs background)
plot_roc(y_true_signal, y_pred_signal, "Classifier_ROC_Signal_vs_BackgroundTest")
plot_sig(y_true_signal, y_pred_signal, "Classifier_SIG_Signal_vs_BackgroundTest")



#########THIS SECTION OF CODE AIMS TO GET THE CLASSIFIER PERFORMANCE ON THE TRAINING DATA, IN TESTLOADER; DOES ALL THE SAME STUFF AS BEFORE##########
y_true_signal = []
y_pred_signal = []

y_true_region = []
y_pred_region_logits = []

with torch.no_grad():
    for x_batch, y_signal, y_region in train_loader:
        logits_signal, coding = classifier(x_batch) #ADV-ED COMMENTED OUT ADDED FEATURE
        probs_signal = torch.sigmoid(logits_signal)
        logits_region = adversary(coding.detach(), lambda_=1.0)#logits_signal.unsqueeze(1)

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
plt.savefig("Adversary_ROC_MulticlassTrainNoMassDiscrFarZ.png")

# Classifier ROC (signal vs background)
plot_roc(y_true_signal, y_pred_signal, "Classifier_ROC_Signal_vs_BackgroundTrain")
plot_sig(y_true_signal, y_pred_signal, "Classifier_SIG_Signal_vs_BackgroundTrain")

import torch
import matplotlib.pyplot as plt
import numpy as np


##################THIS LINE GETS THE SIGMOID VALUE OF THE CODE; IE THE OUTPUT OF THE CLASSIFIER BEFORE THE LAST CLASSIFYING STEP.###########
##################HELPS AS A SANITY CHECK TO SEE IT########

# Assuming `classifier` is your trained model and `data_loader` provides the data
logits_signal = []
logits_background = []

classifier.eval()
with torch.no_grad():
    for x_batch, y_batch, y_region in test_loader:
        logits, coding = classifier(x_batch).squeeze()  # This should be the *pre-sigmoid* value
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
## YB visualizes how separable the pre-sigmoid outputs are for signal vs. background
plt.hist(logits_signal, bins=50, alpha=0.5, label='Signal', density=True, histtype='step')
plt.hist(logits_background, bins=50, alpha=0.5, label='Background', density=True, histtype='step')
plt.xlabel('Classifier Output (Pre-Sigmoid Logit)')
plt.ylabel('Density')
plt.legend()
plt.title('Logit Distribution Before Sigmoid')
plt.grid(True)
plt.savefig("SigmoidBingBongNoMassDiscrFarZ.png")






"""????
    #yb old one recent one
    #RESUMED FOR PRINTING 30 VARIABLES TRAIN AND TEST DOWN: print("Epoch", epoch+1, "total loss=", round(epoch_loss_sum, 4))
    ##for epoch 1 plot per-batch losses
    if epoch == 1:
        plt.figure()
        plt.plot(train_batch_losses)
        plt.xlabel("Training batch index")
        plt.ylabel("Batch training loss")
        plt.title("Epoch 1: Training Loss per Batch")
        plt.savefig("epoch1_train_loss_per_batch.png")
        plt.close()



    ##YB ---- Validation pass - no updates (change in gradient??) -----
    val_batch_losses0 = []
    validate_loss_sum = 0.0
    num_val_batches = 0
    classifier.eval()
    with torch.no_grad():
        for x_val, y_val, _ in test_loader:
        # Step 1: forward through classifier
            val_logits = classifier(x_val)
    
        # Step 2: classification loss
            val_loss_clf = criterion_clf(val_logits, y_val.float())
            val_batch_losses0.append(val_loss_clf.item())
            validate_loss_sum += val_loss_clf.item()
            num_val_batches += 1
        classifier.train()

        avg_val = validate_loss_sum / num_val_batches
        avgValidate_loss_list.append(avg_val)

    classifier.train() #new edit: back to training mode
"""
