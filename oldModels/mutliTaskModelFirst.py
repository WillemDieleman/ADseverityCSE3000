#%%
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from dataset import *
from models import *
from trainAndTest import *


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#%%
# Load the data
microglia = sc.read_h5ad("../data/ROSMAP/microglia.h5ad")

#vasc = sc.read_h5ad("../data/ROSMAP/vascular.niche.h5ad")
metadata = pd.read_csv("../data/ROSMAP/ROSMAP_clinical.csv")
#%%

#%%


# Preprocess the data
def preprocess(annList, metadata):
    # Process metadata and get mappings

    columns = [
        "individualID", "braaksc", "ceradsc", "cogdx", "msex", "age_at_visit_max", "apoe_genotype", "pmi"
    ]

    # columns2 = [
    #     "individualID", "braaksc", "ceradsc", "age_at_visit_max"
    # ]
    # Select and clean relevant metadata
    metadata_clean = metadata[columns].dropna()

    # drop all the columns were cogdx = 6
    metadata_clean = metadata_clean[metadata_clean["cogdx"] != 6]
    #
    # map the cogdx to an integer value
    metadata_clean["cogdx"] = metadata_clean["cogdx"].astype(int) - 1
    #
    # map the ceradsc to an integer value and minus 1
    metadata_clean["ceradsc"] = metadata_clean["ceradsc"].astype(int) - 1

    metadata_clean["ceradsc"] = 3 - metadata_clean["ceradsc"]
    #
    # take only the samples were pmi < 12 hours
    metadata_clean = metadata_clean[metadata_clean["pmi"] < 12]

    # # bin the braaksc to 4 classes, 0-1-2, 3, 4, 5-6
    metadata_clean["braaksc"] = metadata_clean["braaksc"].replace({0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 3})
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
        metadata_clean = metadata_clean[metadata_clean["individualID"].isin(i.obs["individualID"])]

    braak_dict = dict(zip(metadata_clean["individualID"], metadata_clean["braaksc"]))
    cerad_dict = dict(zip(metadata_clean["individualID"], metadata_clean["ceradsc"]))
    cogdx_dict = dict(zip(metadata_clean["individualID"], metadata_clean["cogdx"]))
    sex_dict = dict(zip(metadata_clean["individualID"], metadata_clean["msex"]))
    age_dict = dict(zip(metadata_clean["individualID"], metadata_clean["age_at_visit_max"]))
    apoe_dict = dict(zip(metadata_clean["individualID"], metadata_clean["apoe_genotype"]))

    features = []
    labels = [braak_dict, cerad_dict, cogdx_dict, sex_dict, age_dict, apoe_dict]
    #labels = [braak_dict, cerad_dict, age_dict,]
    datacopies = []
    for data in annList:
        #datacopy = data[data.obs["class"] == 1]
        #datacopy = datacopy[datacopy.obs["state"] == "Mic.2"]

        # limit each braakscore to 5000 samples



        datacopy = data[data.obs["individualID"].isin(metadata_clean["individualID"])]

        #drop all individuals which have 75 or less cells
        counts = datacopy.obs["individualID"].value_counts()
        valid_individuals = counts[counts > 75].index
        datacopy = datacopy[datacopy.obs["individualID"].isin(valid_individuals)]

        # braakData = datacopy.obs["individualID"].map(braak_dict)
        # ceradData = datacopy.obs["individualID"].map(cerad_dict)
        # cogdxData = datacopy.obs["individualID"].map(cogdx_dict)
        # sexData = datacopy.obs["individualID"].map(sex_dict)
        # ageData = datacopy.obs["individualID"].map(age_dict)
        # apoeData = datacopy.obs["individualID"].map(apoe_dict)
        # features = datacopy.X
        datacopies.append(datacopy)

        # labels.append(braakData)
        # labels.append(ceradData)
        # labels.append(cogdxData)
        # labels.append(sexData)
        # labels.append(ageData)
        # labels.append(apoeData)


    return np.array(labels), np.array(features), datacopies

#%%
def sortHVG(input):
    hvg_df = input.var[input.var['highly_variable']].copy()

    # Sort it by the variance (or another metric like dispersion)
    # Use 'variances' or 'dispersions' depending on your settings
    hvg_sorted = hvg_df.sort_values(by='dispersions_norm', ascending=False)

    # Now reorder the AnnData object columns accordingly
    return input[:, hvg_sorted.index].copy()
#%%
# process the data

# sc.pp.highly_variable_genes(microglia, n_top_genes=1000, subset=True)
# microglia = sortHVG(microglia)

scores, _, annListUpdated = preprocess([microglia], metadata)

del microglia
gc.collect()

braakDict = scores[0]
ceradDict = scores[1]
cogdxDict = scores[2]
sexDict = scores[3]
ageDict = scores[4]
apoeDict = scores[5]
# get the indivuals from annListUpdated
individuals = []

for i in annListUpdated:
    individuals = i.obs["individualID"].unique()
individuals = np.array(individuals)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# go through the individuals and get the braakscore for each individual
braak_scores = np.array([braakDict[ind] for ind in individuals])


# # print the division of the braakscore
print("Braakscore distribution:")
print(pd.Series(braak_scores).value_counts())

foldbraakacc, foldbraakpm1acc = [], []
foldbraakCeradacc, foldbraakCeradpm1acc = [], []
foldbraakCogdxacc, foldbraakCogdxpm1acc = [], []
foldbraakSexacc, foldbraakSexpm1acc = [], []
foldbraakAgeacc, foldbraakAgepm1acc = [], []
foldbraakApoeacc, foldbraakApoepm1acc = [], []

# split of a validation set
individuals_main, individuals_val = train_test_split(individuals, test_size=0.2, stratify=braak_scores, random_state=42)
braak_scores_main = np.array([braakDict[ind] for ind in individuals_main])
print("Amount of individuals in main set:", len(individuals_main))
print("Amount of individuals in validation set:", len(individuals_val))


for fold, (train_idx, test_idx) in enumerate(skf.split(individuals_main, braak_scores_main)):
    print(f"Fold {fold + 1}\n-------------------------------")
    if fold != 0:
        break

    individuals_train, individuals_test = individuals_main[train_idx], individuals_main[test_idx]
    X_val = annListUpdated[0][annListUpdated[0].obs["individualID"].isin(individuals_val)]
    y_valBraak = X_val.obs["individualID"].map(braakDict)
    y_valCerad = X_val.obs["individualID"].map(ceradDict)
    y_valCogdx = X_val.obs["individualID"].map(cogdxDict)
    y_valSex = X_val.obs["individualID"].map(sexDict)
    y_valAge = X_val.obs["individualID"].map(ageDict)
    y_valApoc = X_val.obs["individualID"].map(apoeDict)

    # split the individuals into 80/20 train/test
    print("Amount on individuals in train set:", len(individuals_train))
    print("Amount on individuals in test set:", len(individuals_test))
    # Process train and test sets
    X_train = annListUpdated[0][annListUpdated[0].obs["individualID"].isin(individuals_train)]
    X_test = annListUpdated[0][annListUpdated[0].obs["individualID"].isin(individuals_test)]

    y_trainBraak = X_train.obs["individualID"].map(braakDict)
    y_testBraak = X_test.obs["individualID"].map(braakDict)

    y_trainCerad = X_train.obs["individualID"].map(ceradDict)
    y_testCerad = X_test.obs["individualID"].map(ceradDict)

    y_trainCogdx = X_train.obs["individualID"].map(cogdxDict)
    y_testCogdx = X_test.obs["individualID"].map(cogdxDict)

    y_trainSex = X_train.obs["individualID"].map(sexDict)
    y_testSex = X_test.obs["individualID"].map(sexDict)

    y_trainAge = X_train.obs["individualID"].map(ageDict)
    y_testAge = X_test.obs["individualID"].map(ageDict)

    y_trainApoc = X_train.obs["individualID"].map(apoeDict)
    y_testApoc = X_test.obs["individualID"].map(apoeDict)

    individual_X_train = X_train.obs["individualID"]
    individual_X_test = X_test.obs["individualID"]
    individual_X_val = X_val.obs["individualID"]

    # Feature selection
    _, index = selectKBest(X_train.X, y_trainBraak, top_k=1000)
    X_train = X_train[:, index].X
    X_test = X_test[:, index].X
    X_val = X_val[:, index].X

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)


    # print the division of the braakscore
    print("Train set braakscore distribution:")
    dist = pd.Series(y_trainBraak).value_counts()
    print(dist)


    y_train_coral_braak = label_to_coral(y_trainBraak, 4)
    y_test_coral_braak = label_to_coral(y_testBraak, 4)
    y_val_coral_braak = label_to_coral(y_valBraak, 4)
    y_train_coral_cerad = label_to_coral(y_trainCerad, 4)
    y_test_coral_cerad = label_to_coral(y_testCerad, 4)
    y_val_coral_cerad = label_to_coral(y_valCerad, 4)
    y_train_coral_cogdx = label_to_coral(y_trainCogdx, 3)
    y_test_coral_cogdx = label_to_coral(y_testCogdx, 3)
    y_val_coral_cogdx = label_to_coral(y_valCogdx, 3)


    y_trainMTL_cerad = [y_train_coral_braak, y_train_coral_cerad]
    y_testMTL_cerad = [y_test_coral_braak, y_test_coral_cerad]
    y_valMTL_cerad = [y_val_coral_braak, y_val_coral_cerad]

    y_trainMTL_cogdx = [y_train_coral_braak, y_train_coral_cogdx]
    y_testMTL_cogdx = [y_test_coral_braak, y_test_coral_cogdx]
    y_valMTL_cogdx = [y_val_coral_braak, y_val_coral_cogdx]

    y_trainMTL_sex = [y_train_coral_braak, y_trainSex]
    y_testMTL_sex = [y_test_coral_braak, y_testSex]
    y_valMTL_sex = [y_val_coral_braak, y_valSex]

    y_train_MTLage = [y_train_coral_braak, y_trainAge]
    y_test_MTLage = [y_test_coral_braak, y_testAge]
    y_valMTL_age = [y_val_coral_braak, y_valAge]

    y_train_MTLapoe = [y_train_coral_braak, y_trainApoc]
    y_test_MTLapoe = [y_test_coral_braak, y_testApoc]
    y_valMTL_apoe = [y_val_coral_braak, y_valApoc]


    f = torch.float32
    l = torch.long

    # Create dataloaders
    train_loaderBraak_coral, val_loaderBraak_coral, test_loaderBraak_coral = dataLoader(
        X_train, X_val, X_test,
        y_train_coral_braak, y_val_coral_braak, y_test_coral_braak,
        types=[f],
        test_ids=individual_X_test,
        train_ids=individual_X_train,
        val_ids=individual_X_val
    )

    train_loaderMTLcerad, val_loaderMTLcerad, test_loaderMTLcerad = dataLoader(
        X_train, X_val, X_test,
        y_trainMTL_cerad, y_valMTL_cerad, y_testMTL_cerad,
        types=[f, f],
        test_ids=individual_X_test,
        train_ids=individual_X_train,
        val_ids=individual_X_val
    )

    train_loaderMTLcogdx, val_loaderMTLcogdx, test_loaderMTLcogdx = dataLoader(
        X_train, X_val, X_test,
        y_trainMTL_cogdx, y_valMTL_cogdx, y_testMTL_cogdx,
        types=[f, f],
        test_ids=individual_X_test,
        train_ids=individual_X_train,
        val_ids=individual_X_val
    )

    train_loaderMTLsex, val_loaderMTLsex, test_loaderMTLsex = dataLoader(
        X_train, X_val, X_test,
        y_trainMTL_sex, y_valMTL_sex, y_testMTL_sex,
        types=[f, l],
        test_ids=individual_X_test,
        train_ids=individual_X_train,
        val_ids=individual_X_val
    )

    train_loaderMTLAge, val_loaderMTLAge, test_loaderMTLAge = dataLoader(
        X_train, X_val, X_test,
        y_train_MTLage, y_valMTL_age, y_test_MTLage,
        types=[f, f],
        test_ids=individual_X_test,
        train_ids=individual_X_train,
        val_ids=individual_X_val
    )

    train_loaderMTLApoe, val_loaderMTLApoe, test_loaderMTLApoe = dataLoader(
        X_train, X_val, X_test,
        y_train_MTLapoe, y_valMTL_apoe, y_test_MTLapoe,
        types=[f, l],
        test_ids=individual_X_test,
        train_ids=individual_X_train,
        val_ids=individual_X_val
    )


    modelBraak = CoralOrdinalNet(input_dim=1000, num_classes=4)
    modelMTLCerad = MultiTaskModel([4, 4], ["coral", "coral"])
    modelMTLcogdx = MultiTaskModel([4, 3], ["coral", "coral"])
    modelMTLsex = MultiTaskModel([4, 2], ["coral", "classification"])
    modelMTLAge = MultiTaskModel([4, 1], ["coral", "regression"])
    modelMTLApoe = MultiTaskModel([4, 6], ["coral", "classification"])

    loss_fnMSE = nn.MSELoss()
    loss_fnCE = nn.CrossEntropyLoss()

    loss_fnBraak_coral = coral_loss
    loss_fnsCerad = [coral_loss, coral_loss]
    loss_fnsCogdx = [coral_loss, loss_fnMSE]
    loss_fnsSex = [coral_loss, loss_fnCE]
    loss_fnsAge = [coral_loss, loss_fnMSE]
    loss_fnsApoc = [coral_loss, loss_fnCE]

    learningRate = 3e-4
    weightDecay = 0.001


    optimizerBraak = torch.optim.Adam(modelBraak.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerCerad = torch.optim.Adam(modelMTLCerad.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerCogdx = torch.optim.Adam(modelMTLcogdx.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerAge = torch.optim.Adam(modelMTLAge.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerSex = torch.optim.Adam(modelMTLsex.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerApoe = torch.optim.Adam(modelMTLApoe.parameters(), lr=learningRate, weight_decay=weightDecay)

    braakacc, braakpm1acc = [], []
    valLosses = []
    braakCeradacc, braakCeradpm1acc = [], []
    braakCogdxacc, braakCogdxpm1acc = [], []
    braakAgeacc, braakAgepm1acc = [], []
    braakSexacc, braakSexpm1acc = [], []
    braakApoeacc, braakApoepm1acc = [], []


    idsoutput = []
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        print("Training model for Braakscore (coral)")
        train(train_loaderBraak_coral, modelBraak, loss_fnBraak_coral, optimizerBraak)
        _, _, _, idsBraak = test(test_loaderBraak_coral, modelBraak, loss_fnBraak_coral, ordinal=True)
        valLossBraak, _, _, idsBraakVal = test(val_loaderBraak_coral, modelBraak, loss_fnBraak_coral, ordinal=True)

        print("Training multi-task model with Cerad")
        trainMTL(train_loaderMTLcerad, modelMTLCerad, loss_fnsCerad, optimizerCerad)
        _, _, _, _, _, idsCerad = testMTL(test_loaderMTLcerad, modelMTLCerad, loss_fnsCerad, ["coral", "coral"])
        _, _, valLossCerad, _, _, idsCeradVal = testMTL(val_loaderMTLcerad, modelMTLCerad, loss_fnsCerad, ["coral", "coral"])

        print("Training multi-task model with Cogdx")
        trainMTL(train_loaderMTLcogdx, modelMTLcogdx, loss_fnsCogdx, optimizerCogdx)
        _, _, _, _, _, idsCogdx = testMTL(test_loaderMTLcogdx, modelMTLcogdx, loss_fnsCogdx, ["coral", "coral"])
        _, _, valLossCogdx, _, _, idsCogdxVal = testMTL(val_loaderMTLcogdx, modelMTLcogdx, loss_fnsCogdx, ["coral", "coral"])

        print("Training multi-task model with Sex")
        trainMTL(train_loaderMTLsex, modelMTLsex, loss_fnsSex, optimizerSex)
        _, _, _, _, _, idsSex = testMTL(test_loaderMTLsex, modelMTLsex, loss_fnsSex, ["coral", "classification"])
        _, _, valLossSex, _, _, idsSexVal = testMTL(val_loaderMTLsex, modelMTLsex, loss_fnsSex, ["coral", "classification"])

        print("Training multi-task model with Age")
        trainMTL(train_loaderMTLAge, modelMTLAge, loss_fnsAge, optimizerAge)
        _, _, _, _, _, idsAge  = testMTL(test_loaderMTLAge, modelMTLAge, loss_fnsAge, ["coral", "regression"])
        _, _, valLossAge, _, _, idsAgeVal = testMTL(val_loaderMTLAge, modelMTLAge, loss_fnsAge, ["coral", "regression"])

        print("Training multi-task model with Apoe")
        trainMTL(train_loaderMTLApoe, modelMTLApoe, loss_fnsApoc, optimizerApoe)
        _, _, _, _, _, idsApoe = testMTL(test_loaderMTLApoe, modelMTLApoe, loss_fnsApoc, ["coral", "classification"])
        _, _, valLossApoe, _, _, idsApoeVal = testMTL(val_loaderMTLApoe, modelMTLApoe, loss_fnsApoc, ["coral", "classification"])


        valLosses.append({"braak": valLossBraak, "cerad": valLossCerad, "cogdx": valLossCogdx,
                          "sex" : valLossSex, "age": valLossAge, "apoe": valLossApoe})

        val_acc, val_pm1_acc = predictIds(idsBraakVal, braakDict)
        braakacc.append(val_acc), braakpm1acc.append(val_pm1_acc)
        print(f"Braak Validation Accuracy: {val_acc * 100:>0.1f}%, One away: {val_pm1_acc * 100:>0.1f}%")

        val_acc, val_pm1_acc = predictIds(idsCeradVal, braakDict)
        braakCeradacc.append(val_acc), braakCeradpm1acc.append(val_pm1_acc)
        print(f"Cerad Validation Accuracy: {val_acc * 100:>0.1f}%, One away: {val_pm1_acc * 100:>0.1f}%")

        val_acc, val_pm1_acc = predictIds(idsCogdxVal, braakDict)
        braakCogdxacc.append(val_acc), braakCogdxpm1acc.append(val_pm1_acc)
        print(f"Cogdx Validation Accuracy: {val_acc * 100:>0.1f}%, One away: {val_pm1_acc * 100:>0.1f}%")

        val_acc, val_pm1_acc = predictIds(idsSexVal, braakDict)
        braakSexacc.append(val_acc), braakSexpm1acc.append(val_pm1_acc)
        print(f"Sex Validation Accuracy: {val_acc * 100:>0.1f}%, One away: {val_pm1_acc * 100:>0.1f}%")

        val_acc, val_pm1_acc = predictIds(idsAgeVal, braakDict)
        braakAgeacc.append(val_acc), braakAgepm1acc.append(val_pm1_acc)
        print(f"Age Validation Accuracy: {val_acc * 100:>0.1f}%, One away: {val_pm1_acc * 100:>0.1f}%")

        val_acc, val_pm1_acc = predictIds(idsApoeVal, braakDict)
        braakApoeacc.append(val_acc), braakApoepm1acc.append(val_pm1_acc)
        print(f"Apoe Validation Accuracy: {val_acc * 100:>0.1f}%, One away: {val_pm1_acc * 100:>0.1f}%")

        # # write all ids arrays to a file called IDs.txt
        # idsBraakVal = np.array(idsBraakVal)
        # idsCeradVal = np.array(idsCeradVal)
        # idsCogdxVal = np.array(idsCogdxVal)
        # idsSexVal = np.array(idsSexVal)
        # idsAgeVal = np.array(idsAgeVal)
        # idsApoeVal = np.array(idsApoeVal)
        # with open(f"IDs{fold}.txt", "w") as f:
        #     f.write("idsBraak: " + str(idsBraakVal.tolist()) + "\n")
        #     f.write("idsCerad: " + str(idsCeradVal.tolist()) + "\n")
        #     f.write("idsCogdx: " + str(idsCogdxVal.tolist()) + "\n")
        #     f.write("idsSex: " + str(idsSexVal.tolist()) + "\n")
        #     f.write("idsAge: " + str(idsAgeVal.tolist()) + "\n")
        #     f.write("idsApoe: " + str(idsApoeVal.tolist()) + "\n")
        #
        # print("Done!")

    # plot the valLosses
    valLosses = pd.DataFrame(valLosses)
    plt.figure(figsize=(10, 6))
    for col in valLosses.columns:
        plt.plot(valLosses[col], label=col)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss per Task')
    plt.legend()
    plt.savefig(f"valLossesFold{fold}.png")
    plt.show()


    foldbraakacc.append(braakacc)
    foldbraakpm1acc.append(braakpm1acc)
    foldbraakCeradacc.append(braakCeradacc)
    foldbraakCeradpm1acc.append(braakCeradpm1acc)
    foldbraakAgeacc.append(braakAgeacc)
    foldbraakAgepm1acc.append(braakAgepm1acc)
    foldbraakCogdxacc.append(braakCogdxacc)
    foldbraakCogdxpm1acc.append(braakCogdxpm1acc)
    foldbraakSexacc.append(braakSexacc)
    foldbraakSexpm1acc.append(braakSexpm1acc)
    foldbraakApoeacc.append(braakApoeacc)
    foldbraakApoepm1acc.append(braakApoepm1acc)

    # clean up memory

    del X_train, X_test
    del y_trainBraak, y_testBraak
    del y_trainCerad, y_testCerad
    del y_trainCogdx, y_testCogdx
    del y_trainSex, y_testSex
    del y_trainAge, y_testAge
    del y_trainApoc, y_testApoc
    del y_trainMTL_cerad, y_testMTL_cerad
    del y_trainMTL_cogdx, y_testMTL_cogdx
    del y_trainMTL_sex, y_testMTL_sex
    del y_train_MTLage, y_test_MTLage
    del y_train_MTLapoe, y_test_MTLapoe
    del individual_X_test, individual_X_train
    del train_loaderBraak_coral, test_loaderBraak_coral
    del train_loaderMTLcerad, test_loaderMTLcerad
    del train_loaderMTLcogdx, test_loaderMTLcogdx
    del train_loaderMTLAge, test_loaderMTLAge
    del train_loaderMTLsex, test_loaderMTLsex
    del train_loaderMTLApoe, test_loaderMTLApoe
    del modelBraak, modelMTLCerad, modelMTLcogdx, modelMTLsex, modelMTLAge, modelMTLApoe
    del loss_fnBraak_coral, loss_fnsCerad, loss_fnsCogdx, loss_fnsAge, loss_fnsSex, loss_fnsApoc
    del optimizerBraak, optimizerCerad, optimizerCogdx, optimizerAge, optimizerSex, optimizerApoe
    del y_train_coral_braak, y_test_coral_braak
    gc.collect()


foldbraakacc = np.array(foldbraakacc)
foldbraakpm1acc = np.array(foldbraakpm1acc)
foldbraakCeradacc = np.array(foldbraakCeradacc)
foldbraakCeradpm1acc = np.array(foldbraakCeradpm1acc)
foldbraakCogdxacc = np.array(foldbraakCogdxacc)
foldbraakCogdxpm1acc = np.array(foldbraakCogdxpm1acc)
foldbraakSexacc = np.array(foldbraakSexacc)
foldbraakSexpm1acc = np.array(foldbraakSexpm1acc)
foldbraakAgeacc = np.array(foldbraakAgeacc)
foldbraakAgepm1acc = np.array(foldbraakAgepm1acc)
foldbraakApoeacc = np.array(foldbraakApoeacc)
foldbraakApoepm1acc = np.array(foldbraakApoepm1acc)

#export this data to a txt file called outputMTL.txt
with open("../bsc-rp-2425-willem-dieleman/outputMTL.txt", "w") as f:
    for i in range(len(foldbraakacc)):
        f.write(f"Fold {i + 1}\n")
        f.write(f"braakacc: {foldbraakacc[i]}\n")
        f.write(f"braakpm1acc: {foldbraakpm1acc[i]}\n")
        f.write(f"braakCeradacc: {foldbraakCeradacc[i]}\n")
        f.write(f"braakCeradpm1acc: {foldbraakCeradpm1acc[i]}\n")
        f.write(f"braakCogdxacc: {foldbraakCogdxacc[i]}\n")
        f.write(f"braakCogdxpm1acc: {foldbraakCogdxpm1acc[i]}\n")
        f.write(f"braakSexacc: {foldbraakSexacc[i]}\n")
        f.write(f"braakSexpm1acc: {foldbraakSexpm1acc[i]}\n")
        f.write(f"braakAgeacc: {foldbraakAgeacc[i]}\n")
        f.write(f"braakAgepm1acc: {foldbraakAgepm1acc[i]}\n")
        f.write(f"braakApoeacc: {foldbraakApoeacc[i]}\n")
        f.write(f"braakApoepm1acc: {foldbraakApoepm1acc[i]}\n\n")

# average all the result per epoch for all the folds
braakacc = np.mean(foldbraakacc, axis=0)
braakpm1acc = np.mean(foldbraakpm1acc, axis=0)
braakCeradacc = np.mean(foldbraakCeradacc, axis=0)
braakCeradpm1acc = np.mean(foldbraakCeradpm1acc, axis=0)
braakAgeacc = np.mean(foldbraakAgeacc, axis=0)
braakAgepm1acc = np.mean(foldbraakAgepm1acc, axis=0)
braakCogdxacc = np.mean(foldbraakCogdxacc, axis=0)
braakCogdxpm1acc = np.mean(foldbraakCogdxpm1acc, axis=0)
braakSexacc = np.mean(foldbraakSexacc, axis=0)
braakSexpm1acc = np.mean(foldbraakSexpm1acc, axis=0)
braakApoeacc = np.mean(foldbraakApoeacc, axis=0)
braakApoepm1acc = np.mean(foldbraakApoepm1acc, axis=0)


plt.plot(braakacc, label='Braak Accuracy')
plt.plot(braakCeradacc, label='Braak + Cerad Accuracy')
plt.plot(braakCogdxacc, label='Braak + Cogdx Accuracy')
plt.plot(braakAgeacc, label='Braak + Age Accuracy')
plt.plot(braakSexacc, label="Braak + Sex Accuracy")
plt.plot(braakApoeacc, label="Braak + Apoe Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracies with 1k genes")
plt.show()

plt.plot(braakpm1acc, label='Braak Accuracy')
plt.plot(braakCeradpm1acc, label='Braak + Cerad Accuracy')
plt.plot(braakCogdxpm1acc, label='Braak + Cogdx Accuracy')
plt.plot(braakAgepm1acc, label='Braak + Age Accuracy')
plt.plot(braakSexpm1acc, label="Braak + Sex Accuracy")
plt.plot(braakApoepm1acc, label="Braak + Apoe Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracies within 1 stage with 1k genes")
plt.show()

# plot_confusion_matrix(y_testBraak, coral_predict(modelBraak(torch.tensor(X_test, dtype=torch.float32))), classes=[0, 1, 2, 3, 4], title='Confusion Matrix for Braakscore Model')
