import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sympy.physics.units import microgram
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold

# Define the model
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


def train(dataloader, model, loss_fn, optimizer, regression=True):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        if regression:
            loss = loss_fn(pred.squeeze(), y.float())
        else:
            loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()


def test(dataloader, model, loss_fn, regression=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            if regression:
                pred = model(X)
                test_loss += loss_fn(pred.squeeze(), y).item()
                rounded_preds = torch.round(pred).squeeze()
                correct += (rounded_preds == y).float().sum().item()
            else :
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()


    test_loss /= num_batches
    correct /= size
    print(f"Avg loss: {test_loss:>8f}, Correct: {correct*100:>0.1f}%\n")

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

    columns = [
        "individualID", "braaksc", "ceradsc", "cogdx", "msex", "age_at_visit_max", "apoe_genotype"
    ]
    # Select and clean relevant metadata
    metadata_clean = metadata[columns].dropna()

    # drop all the columns were cogdx = 6
    metadata_clean = metadata_clean[metadata_clean["cogdx"] != 6]

    # map the cogdx to an integer value
    metadata_clean["cogdx"] = metadata_clean["cogdx"].astype(int) - 1



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
    additional_inputs = []
    labels = []
    states = []

    # for donor_id in metadata_clean["individualID"]:
    #     donor_cells = []
    #
    #     # Collect all cells from this donor across all AnnData objects
    #     for data in annList:
    #         donor_cells.append(data[data.obs["individualID"] == donor_id].X)
    #
    #     donor_data = donor_cells[0]
    #     for i in range(1, len(donor_cells)):
    #         donor_data = np.concatenate((donor_data, donor_cells[i]), axis=0)
    #
    #     avg = np.mean(donor_data, axis=0)
    #
    #     # Get metadata features and braak class
    #     braak_class = braak_dict[donor_id]
    #     cerad_class = cerad_dict[donor_id]
    #     cogdx_class = cogdx_dict[donor_id]
    #     sex_class = sex_dict[donor_id]
    #     age_class = age_dict[donor_id]
    #     apoe_class = apoe_dict[donor_id]
    #
    #     features.append(avg)
    #     labels.append((braak_class, cerad_class, cogdx_class, sex_class, age_class, apoe_class))

    for data in annList:
        datacopy = data[data.obs["individualID"].isin(metadata_clean["individualID"])]

        braakData = datacopy.obs["individualID"].map(braak_dict)
        ceradData = datacopy.obs["individualID"].map(cerad_dict)
        cogdxData = datacopy.obs["individualID"].map(cogdx_dict)
        sexData = datacopy.obs["individualID"].map(sex_dict)
        ageData = datacopy.obs["individualID"].map(age_dict)
        apoeData = datacopy.obs["individualID"].map(apoe_dict)

        # One-hot encode state labels
        state_ints = datacopy.obs["state"].map(stateDict)
        states_one_hot = pd.get_dummies(state_ints, dtype=int)

        # Ensure all expected state columns are present
        for i in range(len(stateDict)):
            if i not in states_one_hot.columns:
                states_one_hot[i] = 0
        states_one_hot = states_one_hot.sort_index(axis=1)

        features = datacopy.X

        labels.append(braakData)
        labels.append(ceradData)
        labels.append(cogdxData)
        labels.append(sexData)
        labels.append(ageData)
        labels.append(apoeData)
        states.append(states_one_hot.values)


    return np.array(labels), np.array(states), np.array(additional_inputs), np.array(features)

def dataLoader(X_train, X_test, y_train, y_test, type=torch.long):

    train_dataset= TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=type))
    test_dataset= TensorDataset(torch.tensor(X_test, dtype=torch.float32),torch.tensor(y_test, dtype=type))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader




microglia = sc.read_h5ad("../data/ROSMAP/microglia.h5ad")
#vasc = sc.read_h5ad("../data/ROSMAP/vascular.niche.h5ad")
metadata = pd.read_csv("../data/ROSMAP/ROSMAP_clinical.csv")

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
# metadata = pd.read_csv("/tudelft.net/staff-umbrella/bachelorAD/data/ROSMAP/ROSMAP_clinical.csv")
# print("metadata loaded")

print("Preprocessing data...")

# sc.pp.highly_variable_genes(vasc, n_top_genes=1000, subset=True)
sc.pp.highly_variable_genes(microglia, n_top_genes=1000, subset=True)
# sc.pp.highly_variable_genes(inhibitory, n_top_genes=1000, subset=True)
# sc.pp.highly_variable_genes(astrocytes, n_top_genes=1000, subset=True)
# sc.pp.highly_variable_genes(cux2p, n_top_genes=1000, subset=True)
# sc.pp.highly_variable_genes(cux2m, n_top_genes=1000, subset=True)

#scores, additional_inputs, features = preprocess([microglia, inhibitory, astrocytes, cux2p, cux2m], metadata)
scores, states, additional_inputs, features = preprocess([microglia], metadata)

X = features
yBraak = scores[0]
yCerad = scores[1]
yCogdx = scores[2]
ySex = scores[3]
yAge = scores[4]
yApoe = scores[5]
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
    y_trainBraak, y_testBraak = yBraak[train_idx], yBraak[test_idx]
    y_trainCerad, y_testCerad = yCerad[train_idx], yCerad[test_idx]
    y_trainCogdx, y_testCogdx = yCogdx[train_idx], yCogdx[test_idx]
    y_trainSex, y_testSex = ySex[train_idx], ySex[test_idx]
    y_trainAge, y_testAge = yAge[train_idx], yAge[test_idx]
    y_trainApoe, y_testApoe = yApoe[train_idx], yApoe[test_idx]
    y_trainStates, y_testStates = yStates[train_idx], yStates[test_idx]

    # create the train and test dataloaders
    train_loaderBraak, test_loaderBraak = dataLoader(X_train, X_test, y_trainBraak, y_testBraak, type=torch.float32)
    train_loaderCerad, test_loaderCerad = dataLoader(X_train, X_test, y_trainCerad, y_testCerad, type=torch.float32)
    train_loaderCogdx, test_loaderCogdx = dataLoader(X_train, X_test, y_trainCogdx, y_testCogdx, type=torch.float32)
    train_loaderSex, test_loaderSex = dataLoader(X_train, X_test, y_trainSex, y_testSex, type=torch.float32)
    train_loaderAge, test_loaderAge = dataLoader(X_train, X_test, y_trainAge, y_testAge, type=torch.float32)
    train_loaderApoe, test_loaderApoe = dataLoader(X_train, X_test, y_trainApoe, y_testApoe, type=torch.float32)
    train_loaderStates, test_loaderStates = dataLoader(X_train, X_test, y_trainStates, y_testStates, type=torch.float32)

    modelBraak = NeuralNetwork3Layer(1)
    modelCerad = NeuralNetwork3Layer(1)
    modelCogdx = NeuralNetwork3Layer(1)
    modelAge = NeuralNetwork3Layer(1)
    modelSex = NeuralNetwork3Layer(1)
    modelApoe = NeuralNetwork3Layer(1)
    modelStates = NeuralNetwork3Layer(18)

    loss_fnMSE = nn.MSELoss()
    loss_fnCE = nn.CrossEntropyLoss()
    optimizerBraak = torch.optim.Adam(modelBraak.parameters(), lr=1e-3)
    optimizerCerad = torch.optim.Adam(modelCerad.parameters(), lr=1e-3)
    optimizerCogdx = torch.optim.Adam(modelCogdx.parameters(), lr=1e-3)
    optimizerAge = torch.optim.Adam(modelAge.parameters(), lr=1e-3)
    optimizerSex = torch.optim.Adam(modelSex.parameters(), lr=1e-3)
    optimizerApoe = torch.optim.Adam(modelApoe.parameters(), lr=1e-3)
    optimizerStates = torch.optim.SGD(modelStates.parameters(), lr=1e-3)



    lossesBraak, accuracyBraak = [], []
    lossesCerad, accuracyCerad = [], []
    lossesCogdx, accuracyCogdx = [], []
    lossesSex, accuracySex = [], []
    lossesAge, accuracyAge = [], []
    lossesApoe, accuracyApoe = [], []
    lossesStates, accuracyStates = [], []
    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # print("Training model for Braakscore")
        # train(train_loaderBraak, modelBraak, loss_fnMSE, optimizerBraak)
        # a, b = test(test_loaderBraak, modelBraak, loss_fnMSE)
        # lossesBraak.append(a), accuracyBraak.append(b)
        #
        # print("Training model for Ceradscore")
        # train(train_loaderCerad, modelCerad, loss_fnMSE, optimizerCerad)
        # c, d = test(test_loaderCerad, modelCerad, loss_fnMSE)
        # lossesCerad.append(c), accuracyCerad.append(d)
        #
        # print("Training model for Cogdx")
        # train(train_loaderCogdx, modelCogdx, loss_fnMSE, optimizerCogdx)
        # e, f = test(test_loaderCogdx, modelCogdx, loss_fnMSE)
        # lossesCogdx.append(e), accuracyCogdx.append(f)
        #
        # print("Training model for Age")
        # train(train_loaderAge, modelAge, loss_fnMSE, optimizerAge)
        # g, h = test(test_loaderAge, modelAge, loss_fnMSE)
        # lossesAge.append(g), accuracyAge.append(h)
        #
        # print("Training mode for sex")
        # train(train_loaderSex, modelSex, loss_fnMSE, optimizerSex)
        # i, j = test(test_loaderSex, modelSex, loss_fnMSE)
        # lossesSex.append(i), accuracySex.append(j)
        #
        # print("Training model for Apoe")
        # train(train_loaderApoe, modelApoe, loss_fnMSE, optimizerApoe)
        # k, l = test(test_loaderApoe, modelApoe, loss_fnMSE)
        # lossesApoe.append(k), accuracyApoe.append(l)

        print("Training model for states")
        train(train_loaderStates, modelStates, loss_fnCE, optimizerStates, regression=False)
        m, n = test(test_loaderStates, modelStates, loss_fnCE, regression=False)
        lossesStates.append(m), accuracyStates.append(n)

        print("Done!")


    fold_results.append({
        "lossesBraak": lossesBraak,
        "accuracyBraak": accuracyBraak,
        "lossesCerad": lossesCerad,
        "accuracyCerad": accuracyCerad,
        "lossesCogdx": lossesCogdx,
        "accuracyCogdx": accuracyCogdx,
        "lossesSex": lossesSex,
        "accuracySex": accuracySex,
        "lossesAge": lossesAge,
        "accuracyAge": accuracyAge,
        "lossesApoe": lossesApoe,
        "accuracyApoe": accuracyApoe,
        "lossesStates": lossesStates,
        "accuracyStates": accuracyStates
    })



print("Done!")
#export data to a txt file called output

with open("../bsc-rp-2425-willem-dieleman/output.txt", "w") as f:
    i = 0
    for fold in fold_results:
        f.write(f"Fold {i + 1}\n")
        f.write(f"lossesBraak: {fold['lossesBraak']}\n")
        f.write(f"accuracyBraak: {fold['accuracyBraak']}\n")
        f.write(f"lossesCerad: {fold['lossesCerad']}\n")
        f.write(f"accuracyCerad: {fold['accuracyCerad']}\n")
        f.write(f"lossesCogdx: {fold['lossesCogdx']}\n")
        f.write(f"accuracyCogdx: {fold['accuracyCogdx']}\n")
        f.write(f"lossesSex: {fold['lossesSex']}\n")
        f.write(f"accuracySex: {fold['accuracySex']}\n")
        f.write(f"lossesAge: {fold['lossesAge']}\n")
        f.write(f"accuracyAge: {fold['accuracyAge']}\n")
        f.write(f"lossesApoe: {fold['lossesApoe']}\n")
        f.write(f"accuracyApoe: {fold['accuracyApoe']}\n")
        f.write(f"lossesStates: {fold['lossesStates']}\n")
        f.write(f"accuracyStates: {fold['accuracyStates']}\n")
        i += 1


# save the model
#torch.save(model4hidden1.state_dict(), "averageWithMetadata.pth")
#torch.save(model4hidden2.state_dict(), "HighlyVariableGenes.pth")