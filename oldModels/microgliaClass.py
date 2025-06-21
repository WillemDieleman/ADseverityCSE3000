import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

# Define the model
class NeuralNetwork3Layer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


def processMetadata(metadata, annList):
    # List of columns to extract
    # columns = [
    #     "individualID", "braaksc", "age_at_visit_max", "race", "msex",
    #     "educ", "cogdx", "ceradsc", "spanish", "apoe_genotype"
    # ]

    columns = [
        "individualID", "braaksc", "age_at_visit_max", "race", "msex",
        "educ", "spanish", "apoe_genotype"
    ]
    # Select and clean relevant metadata
    metadata_clean = metadata[columns].dropna()

    for i in annList:
        metadata_clean = metadata_clean[metadata_clean["individualID"].isin(i.obs["individualID"])]

    # Map APOE genotypes to numeric categories
    apoe_map = {
        22.0: 1.0,
        23.0: 2.0,
        24.0: 3.0,
        33.0: 4.0,
        34.0: 5.0,
        44.0: 6.0
    }
    metadata_clean["apoe_genotype"] = metadata_clean["apoe_genotype"].map(apoe_map)

    # replace the 90+ sting values in age_at_visit_max with 90.0
    metadata_clean["age_at_visit_max"] = metadata_clean["age_at_visit_max"].replace({"90+": 90.0})
    # replace the strings in age_at_visit_max with their float parsed values
    metadata_clean["age_at_visit_max"] = metadata_clean["age_at_visit_max"].astype(float)

    # normalize the age_at_visit_max
    metadata_clean["age_at_visit_max"] = (metadata_clean["age_at_visit_max"] - metadata_clean[
        "age_at_visit_max"].mean()) / metadata_clean["age_at_visit_max"].std()

    # normalize the educ
    metadata_clean["educ"] = (metadata_clean["educ"] - metadata_clean["educ"].mean()) / metadata_clean["educ"].std()

    # Create lookup dictionaries
    braak_dict = dict(zip(metadata_clean["individualID"], metadata_clean["braaksc"]))
    # extra_dict = metadata_clean.set_index("individualID")[
    #     ["age_at_visit_max", "race", "msex", "educ", "cogdx", "ceradsc", "spanish", "apoe_genotype"]
    # ].to_dict(orient="index")

    extra_dict = metadata_clean.set_index("individualID")[
        ["age_at_visit_max", "race", "msex", "educ", "spanish", "apoe_genotype"]
    ].to_dict(orient="index")

    return extra_dict, braak_dict, metadata_clean


def preprocess(annList, metadata):
    # Process metadata and get mappings

    stateDict = {
        "Mic.1": 0,
        "Mic.2": 1,
        "Mic.3": 2,
        "Mic.4": 3,
        "Mic.5": 4,
        "Mic.6": 5,
        "Mic.7": 6,
        "Mic.8": 7,
        "Mic.9": 8,
        "Mic.10": 9,
        "Mic.11": 10,
        "Mic.12": 11,
        "Mic.13": 12,
        "Mic.14": 13,
        "Mic.15": 14,
        "Mic.16": 15,
        "Macrophages": 16,
        "Monocytes": 17
    }

    features = []
    labels = []
    states = []


    for data in annList:
        # One-hot encode state labels
        state_ints = data.obs["state"].map(stateDict)
        states_one_hot = pd.get_dummies(state_ints, dtype=int)

        # Ensure all expected state columns are present
        for i in range(len(stateDict)):
            if i not in states_one_hot.columns:
                states_one_hot[i] = 0
        states_one_hot = states_one_hot.sort_index(axis=1)

        features = data.X
        states.append(states_one_hot.values)


    return np.array(labels), np.array(states), np.array(features)

def dataLoader(X_train, X_test, y_train, y_test, type=torch.long):

    train_dataset= TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=type))
    test_dataset= TensorDataset(torch.tensor(X_test, dtype=torch.float32),torch.tensor(y_test, dtype=type))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader




microglia = sc.read_h5ad("../data/ROSMAP/microglia.h5ad")
#vasc = sc.read_h5ad("../data/ROSMAP/vascular.niche.h5ad")
metadata = pd.read_csv("../data/ROSMAP/ROSMAP_clinical.csv")


print("Preprocessing data...")

sc.pp.highly_variable_genes(microglia, n_top_genes=1000, subset=True)

#scores, additional_inputs, features = preprocess([microglia, inhibitory, astrocytes, cux2p, cux2m], metadata)
scores, states, features = preprocess([microglia], metadata)

X = features
yStates = states[0]

print("amount of samples: ", len(X))
print("amount of features: ", len(X[0]))


# split the data into train and test sets
#braak data

#5 way cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_trainStates, y_testStates = yStates[train_idx], yStates[test_idx]

    # create the train and test dataloaders
    train_loaderStates, test_loaderStates = dataLoader(X_train, X_test, y_trainStates, y_testStates, type=torch.float32)

    modelStates = NeuralNetwork3Layer(18)

    loss_fnCE = nn.BCEWithLogitsLoss()
    optimizerStates = torch.optim.Adam(modelStates.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizerStates, step_size=5, gamma=0.5)



    lossesStates, accuracyStates = [], []
    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        print("Training model for states")
        train(train_loaderStates, modelStates, loss_fnCE, optimizerStates)
        m, n = test(test_loaderStates, modelStates, loss_fnCE)
        lossesStates.append(m), accuracyStates.append(n)
        scheduler.step()



    fold_results.append({
        "lossesStates": lossesStates,
        "accuracyStates": accuracyStates
    })



print("Done!")
#export data to a txt file called output

with open("../bsc-rp-2425-willem-dieleman/output2.txt", "w") as f:
    i = 0
    for fold in fold_results:
        f.write(f"Fold {i + 1}\n")
        f.write(f"lossesStates: {fold['lossesStates']}\n")
        f.write(f"accuracyStates: {fold['accuracyStates']}\n")
        i += 1


# save the model
#torch.save(model4hidden1.state_dict(), "averageWithMetadata.pth")
#torch.save(model4hidden2.state_dict(), "HighlyVariableGenes.pth")