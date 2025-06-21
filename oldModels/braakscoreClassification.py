import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Define the model
class NeuralNetwork4Hidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(56, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


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

def classify_braak(score):
    if score <= 0:
        return 0
    elif score <= 1:
        return 1
    elif score <= 2:
        return 2
    elif score <= 3:
        return 3
    elif score <= 4:
        return 4
    elif score <= 5:
        return 5
    else:
        return 6

def preprocess(annList, metadata):
    # process the metadata
    extra_dict, braak_dict, metadata_clean = processMetadata(metadata, annList)

    scores = []
    additional_inputs = []
    updatedAnnList = []
    for data in annList:
        datacopy = data[data.obs["individualID"].isin(metadata_clean["individualID"])]

        scoresData = datacopy.obs["individualID"].map(braak_dict)

        additiData = (datacopy.obs["individualID"].map(extra_dict)).apply(lambda x: list(x.values()))
        if len(scores) == 0:
            scores = np.array(scoresData.tolist())
            additional_inputs = np.array(additiData.tolist())
        else:
            class_labels = np.array(scoresData.tolist())
            additiData = np.array(additiData.tolist())
            scores = np.concatenate((scores, class_labels), axis=0)
            additional_inputs = np.concatenate((additional_inputs, additiData), axis=0)

        updatedAnnList.append(datacopy)

    return scores, additional_inputs, updatedAnnList


# microglia = sc.read_h5ad("../data/ROSMAP/microglia.h5ad")
vasc = sc.read_h5ad("../data/ROSMAP/vascular.niche.h5ad")
# print("Loading data...")
# microglia = sc.read_h5ad("/tudelft.net/staff-umbrella/bachelorAD/data/ROSMAP/microglia.h5ad")
# print("Microglia data loaded")
# inhibitory = sc.read_h5ad("/tudelft.net/staff-umbrella/bachelorAD/data/ROSMAP/inhibitory.h5ad")
# print("Inhibitory data loaded")
# astrocytes = sc.read_h5ad("/tudelft.net/staff-umbrella/bachelorAD/data/ROSMAP/astrocytes.h5ad")
# print("Astrocytes data loaded")
# cux2p = sc.read_h5ad("/tudelft.net/staff-umbrella/bachelorAD/data/ROSMAP/cux2+.h5ad")
# print("Cux2+ data loaded")
# cux2m = sc.read_h5ad("/tudelft.net/staff-umbrella/bachelorAD/data/ROSMAP/cux2-.h5ad")
# print("Cux2- data loaded")
print("Gene data loaded")
metadata = pd.read_csv("../data/ROSMAP/ROSMAP_clinical.csv")
#metadata = pd.read_csv("/tudelft.net/staff-umbrella/bachelorAD/data/ROSMAP/ROSMAP_clinical.csv")
print("metadata loaded")

print("Preprocessing data...")
#scores, additional_inputs, updatedAnnList = preprocess([microglia, inhibitory, astrocytes, cux2p, cux2m], metadata)
scores, additional_inputs, updatedAnnList = preprocess([vasc], metadata)
# create the x and y data
X = []
for i in updatedAnnList:
    if len(X) == 0:
        X = i.obsm["X_pca"]
    else:
        X = np.concatenate((X, i.obsm["X_pca"]), axis=0)
# add the extra inputs to the X data
additional_inputs = np.array(additional_inputs.tolist())
X = np.concatenate((X, additional_inputs), axis=1)
y = scores

# create the train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# create the train and test datasets
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
# create the train and test dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model4hidden = NeuralNetwork4Hidden()
print(model4hidden)

loss_fn = nn.MSELoss()
optimizer4hidden = torch.optim.Adam(model4hidden.parameters(), lr=1e-3)

losses = []
epochs = 15
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_loader, model4hidden, loss_fn, optimizer4hidden)
    losses.append(test(test_loader, model4hidden, loss_fn))
print("Done!")
print(losses)