#%%
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
#%%
# Load the data
microglia = sc.read_h5ad("../data/ROSMAP/microglia.h5ad")
#vasc = sc.read_h5ad("../data/ROSMAP/vascular.niche.h5ad")
metadata = pd.read_csv("../data/ROSMAP/ROSMAP_clinical.csv")
#%%
sc.pp.highly_variable_genes(microglia, n_top_genes=1000, subset=True)
#%%
# Preprocess the data
def preprocess(annList, metadata):
    # Process metadata and get mappings

    columns = [
        "individualID", "braaksc", "ceradsc", "cogdx", "msex", "age_at_visit_max", "apoe_genotype"
    ]
    # Select and clean relevant metadata
    metadata_clean = metadata[columns].dropna()

    # drop all the columns were cogdx = 6
    metadata_clean = metadata_clean[metadata_clean["cogdx"] != 6]

    # map the cogdx to an integer value
    metadata_clean["cogdx"] = metadata_clean["cogdx"].astype(int) - 1

    # only take the samples where cogdx is 0 or 3
    metadata_clean = metadata_clean[metadata_clean["cogdx"].isin([0, 3])]

    # map the 0 to 0 and 3 to 1
    metadata_clean["cogdx"] = metadata_clean["cogdx"].replace({0: 0, 3: 1})

    # map the ceradsc to an integer value and minus 1
    metadata_clean["ceradsc"] = metadata_clean["ceradsc"].astype(int) - 1



    # Map APOE genotypes to numeric categories
    apoe_map = {
        22.0: 0,
        23.0: 1,
        24.0: 2,
        33.0: 3,
        34.0: 4,
        44.0: 5
    }
    metadata_clean["apoe_genotype"] = metadata_clean["apoe_genotype"].map(apoe_map)

    # replace the 90+ sting values in age_at_visit_max with 90.0
    metadata_clean["age_at_visit_max"] = metadata_clean["age_at_visit_max"].replace({"90+": 90.0})
    # replace the strings in age_at_visit_max with their float parsed values
    metadata_clean["age_at_visit_max"] = metadata_clean["age_at_visit_max"].astype(float)
    # normalize the age_at_visit_max
    metadata_clean["age_at_visit_max"] = (metadata_clean["age_at_visit_max"] - metadata_clean[
        "age_at_visit_max"].mean()) / metadata_clean["age_at_visit_max"].std()



    for i in annList:
        metadata_clean = metadata_clean[metadata_clean["individualID"].isin(i.obs["individualID"])]

    braak_dict = dict(zip(metadata_clean["individualID"], metadata_clean["braaksc"]))
    cerad_dict = dict(zip(metadata_clean["individualID"], metadata_clean["ceradsc"]))
    cogdx_dict = dict(zip(metadata_clean["individualID"], metadata_clean["cogdx"]))
    sex_dict = dict(zip(metadata_clean["individualID"], metadata_clean["msex"]))
    age_dict = dict(zip(metadata_clean["individualID"], metadata_clean["age_at_visit_max"]))
    apoe_dict = dict(zip(metadata_clean["individualID"], metadata_clean["apoe_genotype"]))

    features = []
    labels = []
    for data in annList:
        datacopy = data[data.obs["individualID"].isin(metadata_clean["individualID"])]

        braakData = datacopy.obs["individualID"].map(braak_dict)
        ceradData = datacopy.obs["individualID"].map(cerad_dict)
        cogdxData = datacopy.obs["individualID"].map(cogdx_dict)
        sexData = datacopy.obs["individualID"].map(sex_dict)
        ageData = datacopy.obs["individualID"].map(age_dict)
        apoeData = datacopy.obs["individualID"].map(apoe_dict)
        features = datacopy.X

        labels.append(braakData)
        labels.append(ceradData)
        labels.append(cogdxData)
        labels.append(sexData)
        labels.append(ageData)
        labels.append(apoeData)


    return np.array(labels), np.array(features)
#%%
#define the models
#single task model
class NeuralNetwork3Layer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#multi task model
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_list):
        super(MultiTaskModel, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.task_layers = nn.ModuleList([
            nn.Linear(64, num_classes) for num_classes in num_classes_list
        ])

    def forward(self, x):
        shared_representation = self.shared_layer(x)
        task_outputs = [task_layer(shared_representation) for task_layer in self.task_layers]
        return task_outputs
#%%
def train(dataloader, model, loss_fn, optimizer, regression=True):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        pred = pred.squeeze()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn, regression=True):
    model.eval()
    test_loss, correct = 0, 0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred = pred.squeeze()
            test_loss += loss_fn(pred, y).item()
            rounded_preds = torch.round(pred).squeeze()
            correct += (rounded_preds == y).float().sum().item()
                
                
    test_loss /= num_batches
    correct /= size
    print(f"Avg loss: {test_loss:>8f}, Correct: {correct*100:>0.1f}%\n")

    return test_loss, correct

def trainMTL(dataloader, model, loss_fns, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        pred = [p.squeeze() for p in pred]
        loss = 0
        for i in range(len(pred)):
            loss += loss_fns[i](pred[i], y[:, i])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def testMTL(dataloader, model, loss_fns):
    model.eval()
    test_loss, correct = 0, 0
    lossArr = np.zeros(len(loss_fns))
    accArr = np.zeros(len(loss_fns))
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred = [p.squeeze() for p in pred]
            loss = 0
            batch_correct = 0
            for i in range(len(pred)):
                temp_loss = loss_fns[i](pred[i], y[:, i])
                loss += temp_loss
                lossArr[i] += temp_loss.item()

                pred_class = pred[i]
                rounded_preds = torch.round(pred_class).squeeze()
                correct_task = (rounded_preds == y[:, i]).float().sum().item()
                accArr[i] += correct_task
                batch_correct += correct_task

            test_loss += loss.item()
            correct += batch_correct / len(pred)  # average across tasks

    test_loss /= num_batches
    for i in range(len(lossArr)):
        lossArr[i] /= num_batches
        accArr[i] /= size
        print(f"Avg loss task {i}: {lossArr[i]:>8f}, Correct task {i}: {accArr[i] * 100:>0.1f}%")
    correct /= size
    print(f"Avg loss: {test_loss:>8f}, Correct: {correct * 100:>0.1f}%\n")

    return test_loss, correct, lossArr, accArr
#%%
def dataLoader(X_train, X_test, y_train, y_test, type=torch.long):

    train_dataset= TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=type))
    test_dataset= TensorDataset(torch.tensor(X_test, dtype=torch.float32),torch.tensor(y_test, dtype=type))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader
#%%
# process the data

annList = [microglia]

scores, features = preprocess(annList, metadata)

# Convert to PyTorch tensors
X = features
yBraak = scores[0].astype(int)
yCerad = scores[1].astype(int)
yCogdx = scores[2].astype(int)
ySex = scores[3].astype(int)
yAge = scores[4].astype(float)
yApoe = scores[5].astype(int)




#%%

# Perform a single 80/20 split
X_train, X_test, y_trainBraak, y_testBraak = train_test_split(
    X, yBraak, test_size=0.2, random_state=42
)
_, _, y_trainCerad, y_testCerad = train_test_split(
    X, yCerad, test_size=0.2, random_state=42
)
_, _, y_trainCogdx, y_testCogdx = train_test_split(
    X, yCogdx, test_size=0.2, random_state=42
)
_, _, y_trainSex, y_testSex = train_test_split(
    X, ySex, test_size=0.2, random_state=42
)
_, _, y_trainAge, y_testAge = train_test_split(
    X, yAge, test_size=0.2, random_state=42
)
_, _, y_trainApoe, y_testApoe = train_test_split(
    X, yApoe, test_size=0.2, random_state=42
)


# Multi-task labels
y_trainBCG = np.transpose(np.vstack([y_trainBraak, y_trainCerad, y_trainCogdx]))
y_testBCG = np.transpose(np.vstack([y_testBraak, y_testCerad, y_testCogdx]))

y_trainBAS = np.transpose(np.vstack([y_trainBraak, y_trainAge, y_trainSex]))
y_testBAS = np.transpose(np.vstack([y_testBraak, y_testAge, y_testSex]))

y_trainAll = np.transpose(np.vstack([y_trainBraak, y_trainCerad, y_trainCogdx, y_trainAge, y_trainSex, y_trainApoe]))
y_testAll = np.transpose(np.vstack([y_testBraak, y_testCerad, y_testCogdx, y_testAge, y_testSex, y_testApoe]))

# Create dataloaders
train_loaderBraak, test_loaderBraak = dataLoader(X_train, X_test, y_trainBraak, y_testBraak, type=torch.float32)
train_loaderBCG, test_loaderBCG = dataLoader(X_train, X_test, y_trainBCG, y_testBCG, type=torch.float32)
train_loaderBAS, test_loaderBAS = dataLoader(X_train, X_test, y_trainBAS, y_testBAS, type=torch.float32)
train_loaderAll, test_loaderAll = dataLoader(X_train, X_test, y_trainAll, y_testAll, type=torch.float32)
    

modelBraak = NeuralNetwork3Layer(1)
modelBCG = MultiTaskModel([1, 1, 1])
modelBAS = MultiTaskModel([1, 1, 1])
modelAll = MultiTaskModel([1, 1, 1, 1, 1, 1])

loss_fnMSE = nn.MSELoss()


optimizerBraak = torch.optim.Adam(modelBraak.parameters(), lr=1e-3)
optimizerMTL = torch.optim.Adam(modelBCG.parameters(), lr=1e-3)
optimizerBAS = torch.optim.Adam(modelBAS.parameters(), lr=1e-3)
optimizerAll = torch.optim.Adam(modelAll.parameters(), lr=1e-3)

lossesBraak, accuracyBraak = [], []
lossesBCG, accuracyBCG, lossArrBCG, accArrBCG = [], [], [], []
lossesBAS, accuracyBAS, lossArrBAS, accArrBAS = [], [], [], []
lossesAll, accuracyAll, lossArrAll, accArrAll = [], [], [], []
epochs = 15
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    print("Training model for Braakscore")
    train(train_loaderBraak, modelBraak, loss_fnMSE, optimizerBraak)
    a, b = test(test_loaderBraak, modelBraak, loss_fnMSE)
    lossesBraak.append(a), accuracyBraak.append(b)

    loss_fns = [loss_fnMSE, loss_fnMSE, loss_fnMSE]
    print("Training multi-task model: Braak, Cerad, Cogdx")
    trainMTL(train_loaderBCG, modelBCG, loss_fns, optimizerMTL)
    e, f, g, h = testMTL(test_loaderBCG, modelBCG, loss_fns)
    lossesBCG.append(e), accuracyBCG.append(f), lossArrBCG.append(g), accArrBCG.append(h)

    print("Training multi-task model: Braak, Cerad, Cogdx")
    trainMTL(train_loaderBAS, modelBAS, loss_fns, optimizerBAS)
    i, j, k, l = testMTL(test_loaderBAS, modelBAS, loss_fns)
    lossesBAS.append(e), accuracyBAS.append(f), lossArrBAS.append(g), accArrBAS.append(h)

    loss_fns2 = [loss_fnMSE, loss_fnMSE, loss_fnMSE, loss_fnMSE, loss_fnMSE, loss_fnMSE]

    print("Training multi-task model: All")
    trainMTL(train_loaderAll, modelAll, loss_fns2, optimizerAll)
    m, n, o, p = testMTL(test_loaderAll, modelAll, loss_fns2)
    lossesAll.append(m), accuracyAll.append(n), lossArrAll.append(o), accArrAll.append(p)

    print("Done!")

# export data to a txt file called outputMTL
with open("../bsc-rp-2425-willem-dieleman/outputMTL.txt", "w") as f:
    f.write(f"lossesMTL: {lossesBCG}\n")
    f.write(f"accuracyMTL: {accuracyBCG}\n")
    f.write(f"lossArr: {lossArrBCG}\n")
    f.write(f"accArr: {accArrBCG}\n")

    f.write(f"lossesBraak: {lossesBraak}\n")
    f.write(f"accuracyBraak: {accuracyBraak}\n")


# create confusion matrix for the braakscore test set
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_predBraak = []
y_trueBraak = []
with torch.no_grad():
    for X, y in test_loaderBraak:
        pred = modelBraak(X)
        pred = pred.squeeze()
        y_predBraak.append(pred.numpy())
        y_trueBraak.append(y.numpy())
y_predBraak = np.concatenate(y_predBraak)
y_trueBraak = np.concatenate(y_trueBraak)
y_predBraak = np.round(y_predBraak).astype(int)
cm = confusion_matrix(y_trueBraak, y_predBraak)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Braakscore')
plt.savefig("confusion_matrix_Braakscore.png")




# export the models to analyze
torch.save(modelBraak.state_dict(), "modelBraak.pt")
torch.save(modelBCG.state_dict(), "modelBCG.pt")
torch.save(modelBAS.state_dict(), "modelBAS.pt")
torch.save(modelAll.state_dict(), "modelAll.pt")
