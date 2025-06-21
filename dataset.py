from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.feature_selection import f_classif, SelectKBest
import pandas as pd

class CustomTestDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = to_tensor_safe(X, torch.float32)
        self.y = to_tensor_safe(y, torch.float32)
        self.ids = ids  # keep as list/array of strings

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ids.iloc[idx]

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_list, ids, types):
        self.X = to_tensor_safe(X, torch.float32)
        self.y_list = [to_tensor_safe(y, dtype) for y, dtype in zip(y_list, types)]
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], [y[idx] for y in self.y_list], self.ids.iloc[idx]

def dataLoader(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    types,
    test_ids=None, train_ids=None, val_ids=None
):
    if isinstance(y_train, (list, tuple)) and isinstance(y_test, (list, tuple)):
        # Multi-task dataset
        train_dataset = MultiTaskDataset(X_train, y_train, ids=train_ids, types=types)
        val_dataset = MultiTaskDataset(X_val, y_val, ids=val_ids, types=types)
        test_dataset = MultiTaskDataset(X_test, y_test, ids=test_ids, types=types)
    else:
        # Single-task dataset
        train_dataset = TensorDataset(
            to_tensor_safe(X_train, torch.float32),
            to_tensor_safe(y_train, types[0])
        )
        if test_ids is not None:
            test_dataset = CustomTestDataset(X_test, y_test, test_ids)
            val_dataset = CustomTestDataset(X_val, y_val, val_ids)
        else:
            test_dataset = TensorDataset(
                to_tensor_safe(X_test, torch.float32),
                to_tensor_safe(y_test, types[0])
            )
            val_dataset = TensorDataset(
                to_tensor_safe(X_val, torch.float32),
                to_tensor_safe(y_val, types[0])
            )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

def select_features_anova(X, y, top_k=1000):
    f_values, p_values = f_classif(X, y)

    # Get indices of top_k features based on F-values
    selected_indices = np.argsort(f_values)[-top_k:]

    # Subset the feature matrix to include only the selected features
    X_selected = X[:, selected_indices]

    return X_selected, selected_indices


def select_top_features_multi_task(X, y_list, top_k=1000, agg_method="sum"):
    """
    Select top_k features based on multi-task univariate feature selection.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y_list (List[np.ndarray]): List of label arrays, one per task
        top_k (int): Number of features to select
        agg_method (str): Method to aggregate scores: "sum", "max", or "mean"

    Returns:
        selected_indices (np.ndarray): Indices of selected features
    """
    all_scores = []

    for y in y_list:
        # Skip regression targets, which are not suitable for f_classif
        if len(np.unique(y)) > 20:  # crude check to skip continuous labels
            continue

        scores, _ = f_classif(X, y)
        all_scores.append(scores)

    if not all_scores:
        raise ValueError("No suitable tasks for ANOVA F-test.")

    # Stack scores and aggregate
    score_matrix = np.vstack(all_scores)  # shape: (n_tasks, n_features)

    if agg_method == "sum":
        agg_scores = score_matrix.sum(axis=0)
    elif agg_method == "max":
        agg_scores = score_matrix.max(axis=0)
    elif agg_method == "mean":
        agg_scores = score_matrix.mean(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")

    # Select top_k feature indices
    selected_indices = np.argsort(agg_scores)[-top_k:]
    return selected_indices

def selectKBest(X, y, top_k=1000):
    selector = SelectKBest(f_classif, k=top_k)
    selector.fit(X, y)

    # Get indices of top k features based on F-values
    cols_anova = selector.get_support(indices=True)

    # Subset the feature matrix to include only the selected features
    X_selected = X[:, cols_anova]

    return X_selected, cols_anova

def to_tensor_safe(val, dtype):
    if isinstance(val, torch.Tensor):
        return val.detach().clone().to(dtype)
    else:
        if isinstance(val, pd.Series):
            val = val.values
        return torch.tensor(val, dtype=dtype)

def preprocess(annList, metadata, embeddings=False):
    if not embeddings:
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

        for data in annList:
            # One-hot encode state labels
            data.obs["state"] = data.obs["state"].map(stateDict)

            #only keep the cells that are in the ceradTypes


    # Process metadata and get mappings

    columns = [
        "individualID", "braaksc", "ceradsc", "cogdx", "msex", "age_at_visit_max", "apoe_genotype", "pmi"
    ]

    # columns2 = [
    #     "individualID", "braaksc", "ceradsc", "age_at_visit_max"
    # ]
    # Select and clean relevant metadata
    metadata_clean = metadata[columns].dropna()

    labels = []

    for ind in metadata_clean["individualID"]:
        cogdx = metadata_clean[metadata_clean["individualID"] == ind]["cogdx"].values[0]
        braaksc = metadata_clean[metadata_clean["individualID"] == ind]["braaksc"].values[0]
        ceradsc = metadata_clean[metadata_clean["individualID"] == ind]["ceradsc"].values[0]

        if cogdx == 1 and braaksc <= 3 and ceradsc >= 3:
            label = "CT"
        elif cogdx == 4 and braaksc >= 4 and ceradsc <= 2:
            label = "AD"
        else:
            label = "Other"

        labels.append(label)
    metadata_clean["AD"] = labels
    print("AD labels:", metadata_clean["AD"].value_counts())

    # metadata_clean = metadata_clean[metadata_clean["AD"] != "Other"]
    #
    # metadata_clean["AD"] = metadata_clean["AD"].map({"CT": 0, "AD": 1})

    # drop all the columns were cogdx = 6
    metadata_clean = metadata_clean[metadata_clean["cogdx"] != 6]
    #
    # map the cogdx to an integer value
    metadata_clean["cogdx"] = metadata_clean["cogdx"].astype(int) - 1
    #
    # map the ceradsc to an integer value and minus 1


    # metadata_clean = metadata_clean[metadata_clean["ceradsc"] != 2]
    # metadata_clean = metadata_clean[metadata_clean["ceradsc"] != 3]
    # metadata_clean["ceradsc"] = metadata_clean["ceradsc"].replace({1: 0, 4: 1})

    # metadata_clean = metadata_clean[metadata_clean["cogdx"] != 2]
    # metadata_clean = metadata_clean[metadata_clean["cogdx"] != 3]
    # metadata_clean = metadata_clean[metadata_clean["cogdx"] != 5]
    # metadata_clean["cogdx"] = metadata_clean["cogdx"].replace({1: 0, 4: 1})


    metadata_clean["ceradsc"] = metadata_clean["ceradsc"].astype(int) - 1

    #metadata_clean["ceradsc"] = 3 - metadata_clean["ceradsc"]
    #
    # take only the samples were pmi < 12 hours
    metadata_clean = metadata_clean[metadata_clean["pmi"] < 12]

    # # bin the braaksc to 4 classes, 0-1-2, 3, 4, 5-6
    metadata_clean["braaksc"] = metadata_clean["braaksc"].replace({0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 3})
    #metadata_clean["ceradsc"] = metadata_clean["ceradsc"].replace({0: 0, 1: 1, 2: 2, 3: 2})
    metadata_clean["cogdx"] = metadata_clean["cogdx"].replace({0: 0, 1: 1, 2: 1, 3: 2, 4: 2})



    # # Map APOE genotypes to numeric categories
    apoe_map = {
        22.0: 0,
        23.0: 1,
        24.0: 2,
        33.0: 3,
        34.0: 4,
        44.0: 5
    }



    metadata_clean["apoe_genotype"] = metadata_clean["apoe_genotype"].map(apoe_map)
    #
    # replace the 90+ sting values in age_at_visit_max with 90.0
    metadata_clean["age_at_visit_max"] = metadata_clean["age_at_visit_max"].replace({"90+": 90.0})
    # replace the strings in age_at_visit_max with their float parsed values
    metadata_clean["age_at_visit_max"] = metadata_clean["age_at_visit_max"].astype(float)
    # normalize the age_at_visit_max
    metadata_clean["age_at_visit_max"] = (metadata_clean["age_at_visit_max"] - metadata_clean[
        "age_at_visit_max"].mean()) / metadata_clean["age_at_visit_max"].std()



    for i in annList:
        if embeddings:
            metadata_clean = metadata_clean[metadata_clean["individualID"].isin(i["individualID"])]
        else:
            metadata_clean = metadata_clean[metadata_clean["individualID"].isin(i.obs["individualID"])]

    braak_dict = dict(zip(metadata_clean["individualID"], metadata_clean["braaksc"]))
    cerad_dict = dict(zip(metadata_clean["individualID"], metadata_clean["ceradsc"]))
    cogdx_dict = dict(zip(metadata_clean["individualID"], metadata_clean["cogdx"]))
    sex_dict = dict(zip(metadata_clean["individualID"], metadata_clean["msex"]))
    age_dict = dict(zip(metadata_clean["individualID"], metadata_clean["age_at_visit_max"]))
    apoe_dict = dict(zip(metadata_clean["individualID"], metadata_clean["apoe_genotype"]))
    AD_dict = dict(zip(metadata_clean["individualID"], metadata_clean["AD"]))

    features = []
    labels = [braak_dict, cerad_dict, cogdx_dict, sex_dict, age_dict, apoe_dict, AD_dict]
    #labels = [braak_dict, cerad_dict, age_dict,]
    datacopies = []
    for data in annList:
        #datacopy = data[data.obs["class"] == 1]
        #datacopy = datacopy[datacopy.obs["state"] == "Mic.2"]
        if embeddings:
            datacopy = data[data["individualID"].isin(metadata_clean["individualID"])]
        else:
            #types = [1, 11, 12, 13, 14] # cerad types
            #types = [2, 3, 4, 10, 11, 12, 15] # braak types
            #types = [10, 11, 12, 14] # cogdx types
            #data = data[data.obs["state"].isin(types)]
            datacopy = data[data.obs["individualID"].isin(metadata_clean["individualID"])]
            #counts = datacopy.obs["individualID"].value_counts()
            #valid_individuals = counts[counts > 20].index
            #datacopy = datacopy[datacopy.obs["individualID"].isin(valid_individuals)]
            #datacopy = datacopy[datacopy.obs["DoubletFinder.score"] < 0.5] # filter out doublets


        #drop all individuals which have 75 or less cells
        # counts = datacopy.obs["individualID"].value_counts()
        # valid_individuals = counts[counts > 75].index
        # datacopy = datacopy[datacopy.obs["individualID"].isin(valid_individuals)]
        datacopies.append(datacopy)


    return np.array(labels), np.array(features), datacopies