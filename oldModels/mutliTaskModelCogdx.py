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

    #metadata_clean["ceradsc"] = 3 - metadata_clean["ceradsc"]
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
Cogdx_scores = np.array([cogdxDict[ind] for ind in individuals])


# # print the division of the braakscore
print("Cogdx score distribution:")
print(pd.Series(Cogdx_scores).value_counts())

foldResults = []

# foldCERADacc, foldCERADpm1acc = [], []
# foldCERADBraakacc, foldCERADBraakpm1acc = [], []
# foldCERADCogdxacc, foldCERADCogdxpm1acc = [], []
# foldCERADSexacc, foldCERADSexpm1acc = [], []
# foldCERADAgeacc, foldCERADAgepm1acc = [], []
# foldCERADApoeacc, foldCERADApoepm1acc = [], []

# split of a validation set
individuals_main, individuals_val = train_test_split(individuals, test_size=0.2, stratify=Cogdx_scores, random_state=42)
Cogdx_scores_main = np.array([cogdxDict[ind] for ind in individuals_main])
print("Amount of individuals in main set:", len(individuals_main))
print("Amount of individuals in validation set:", len(individuals_val))


for fold, (train_idx, test_idx) in enumerate(skf.split(individuals_main, Cogdx_scores_main)):
    print(f"Fold {fold + 1}\n-------------------------------")

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
    _, index = selectKBest(X_train.X, y_trainCogdx, top_k=1000)
    X_train = X_train[:, index].X
    X_test = X_test[:, index].X
    X_val = X_val[:, index].X

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)


    # print the division of the braakscore
    print("Train set Cogdx score distribution:")
    dist = pd.Series(y_trainCogdx).value_counts()
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


    y_trainMTL_braak = [y_train_coral_cogdx, y_train_coral_braak]
    y_testMTL_braak = [y_test_coral_cogdx, y_test_coral_braak]
    y_valMTL_braak = [y_val_coral_cogdx, y_val_coral_braak]

    y_trainMTL_CERAD = [y_train_coral_cogdx, y_train_coral_cerad]
    y_testMTL_CERAD = [y_test_coral_cogdx, y_test_coral_cerad]
    y_valMTL_CERAD = [y_val_coral_cogdx, y_val_coral_cerad]

    y_trainMTL_sex = [y_train_coral_cogdx, y_trainSex]
    y_testMTL_sex = [y_test_coral_cogdx, y_testSex]
    y_valMTL_sex = [y_val_coral_cogdx, y_valSex]

    y_train_MTLage = [y_train_coral_cogdx, y_trainAge]
    y_test_MTLage = [y_test_coral_cogdx, y_testAge]
    y_valMTL_age = [y_val_coral_cogdx, y_valAge]

    y_train_MTLapoe = [y_train_coral_cogdx, y_trainApoc]
    y_test_MTLapoe = [y_test_coral_cogdx, y_testApoc]
    y_valMTL_apoe = [y_val_coral_cogdx, y_valApoc]


    f = torch.float32
    l = torch.long

    # Create dataloaders
    train_loaderCogdx_coral, val_loaderCogdx_coral, test_loaderCogdxcoral = dataLoader(
        X_train, X_val, X_test,
        y_train_coral_cogdx, y_val_coral_cogdx, y_test_coral_cogdx,
        types=[f],
        test_ids=individual_X_test,
        train_ids=individual_X_train,
        val_ids=individual_X_val
    )

    train_loaderMTLbraak, val_loaderMTLbraak, test_loaderMTLbraak = dataLoader(
        X_train, X_val, X_test,
        y_trainMTL_braak, y_valMTL_braak, y_testMTL_braak,
        types=[f, f],
        test_ids=individual_X_test,
        train_ids=individual_X_train,
        val_ids=individual_X_val
    )

    train_loaderMTLCERAD, val_loaderMTLCERAD, test_loaderMTLCERAD = dataLoader(
        X_train, X_val, X_test,
        y_trainMTL_CERAD, y_valMTL_CERAD, y_testMTL_CERAD,
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


    modelCogdx = CoralOrdinalNet(input_dim=1000, num_classes=3)
    modelMTLBraak = MultiTaskModel([3, 4], ["coral", "coral"])
    modelMTLCERAD = MultiTaskModel([3, 4], ["coral", "coral"])
    modelMTLsex = MultiTaskModel([3, 2], ["coral", "classification"])
    modelMTLAge = MultiTaskModel([3, 1], ["coral", "regression"])
    modelMTLApoe = MultiTaskModel([3, 6], ["coral", "classification"])

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


    optimizerBraak = torch.optim.Adam(modelCogdx.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerCerad = torch.optim.Adam(modelMTLBraak.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerCogdx = torch.optim.Adam(modelMTLCERAD.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerAge = torch.optim.Adam(modelMTLAge.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerSex = torch.optim.Adam(modelMTLsex.parameters(), lr=learningRate, weight_decay=weightDecay)
    optimizerApoe = torch.optim.Adam(modelMTLApoe.parameters(), lr=learningRate, weight_decay=weightDecay)

    CERADacc, CERADpm1acc = [], []
    valLosses = []
    accs = []


    idsoutput = []
    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        print("Training model for Cogdx score (coral)")
        train(train_loaderCogdx_coral, modelCogdx, loss_fnBraak_coral, optimizerBraak)
        _, _, _, idsCogdx = test(test_loaderCogdxcoral, modelCogdx, loss_fnBraak_coral, ordinal=True)
        valLossCogdx, _, _, idsCogdxVal = test(val_loaderCogdx_coral, modelCogdx, loss_fnBraak_coral, ordinal=True)

        print("Training multi-task model with Braak")
        trainMTL(train_loaderMTLbraak, modelMTLBraak, loss_fnsCerad, optimizerCerad)
        _, _, _, _, _, idsbraak = testMTL(test_loaderMTLbraak, modelMTLBraak, loss_fnsCerad, ["coral", "coral"])
        _, _, valLossBraak, _, _, idsBraakVal = testMTL(val_loaderMTLbraak, modelMTLBraak, loss_fnsCerad, ["coral", "coral"])

        print("Training multi-task model with CERAD")
        trainMTL(train_loaderMTLCERAD, modelMTLCERAD, loss_fnsCogdx, optimizerCogdx)
        _, _, _, _, _, idsCERAD = testMTL(test_loaderMTLCERAD, modelMTLCERAD, loss_fnsCogdx, ["coral", "coral"])
        _, _, valLossCERAD, _, _, idsCERADVal = testMTL(val_loaderMTLCERAD, modelMTLCERAD, loss_fnsCogdx, ["coral", "coral"])

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


        valLosses.append({"braak": valLossBraak, "CERAD": valLossCERAD, "cogdx": valLossCogdx,
                          "sex" : valLossSex, "age": valLossAge, "apoe": valLossApoe})

        IMPORTANTDICT = cogdxDict

        Cogdxval_acc, Cogdxval_pm1_acc = predictIds(idsCogdxVal, IMPORTANTDICT)
        print(f"Cogdx Validation Accuracy: {Cogdxval_acc * 100:>0.1f}%, One away: {Cogdxval_pm1_acc * 100:>0.1f}%")

        CERADval_acc, CERADval_pm1_acc = predictIds(idsCERADVal, IMPORTANTDICT)
        print(f"Cerad Validation Accuracy: {CERADval_acc * 100:>0.1f}%, One away: {CERADval_pm1_acc * 100:>0.1f}%")

        Braakval_acc, Braakval_pm1_acc = predictIds(idsBraakVal, IMPORTANTDICT)
        print(f"Braak Validation Accuracy: {Braakval_acc * 100:>0.1f}%, One away: {Braakval_pm1_acc * 100:>0.1f}%")

        Sexval_acc, Sexval_pm1_acc = predictIds(idsSexVal, IMPORTANTDICT)
        print(f"Sex Validation Accuracy: {Sexval_acc * 100:>0.1f}%, One away: {Sexval_pm1_acc * 100:>0.1f}%")

        Ageval_acc, Ageval_pm1_acc = predictIds(idsAgeVal, IMPORTANTDICT)
        print(f"Age Validation Accuracy: {Ageval_acc * 100:>0.1f}%, One away: {Ageval_pm1_acc * 100:>0.1f}%")

        APOEval_acc, APOEval_pm1_acc = predictIds(idsApoeVal, IMPORTANTDICT)
        print(f"Apoe Validation Accuracy: {APOEval_acc * 100:>0.1f}%, One away: {APOEval_pm1_acc * 100:>0.1f}%")

        accs.append({
            "CERAD": CERADval_acc, "CERADpm1": CERADval_pm1_acc,
            "Braak": Braakval_acc, "Braakpm1": Braakval_pm1_acc,
            "Cogdx": Cogdxval_acc, "Cogdxpm1": Cogdxval_pm1_acc,
            "Sex": Sexval_acc, "Sexpm1": Sexval_pm1_acc,
            "Age": Ageval_acc, "Agepm1": Ageval_pm1_acc,
            "Apoe": APOEval_acc, "Apoepm1": APOEval_pm1_acc
        })

        # write all ids arrays to a file called IDs.txt
        idsCERADVal = np.array(idsCERADVal)
        idsBraakVal = np.array(idsBraakVal)
        idsCogdxVal = np.array(idsCogdxVal)
        idsSexVal = np.array(idsSexVal)
        idsAgeVal = np.array(idsAgeVal)
        idsApoeVal = np.array(idsApoeVal)
        with open(f"CogdxIds/IDs{fold}", "w") as f:
            f.write("idsCerad: " + str(idsCERADVal.tolist()) + "\n")
            f.write("idsBraak: " + str(idsBraakVal.tolist()) + "\n")
            f.write("idsCogdx: " + str(idsCogdxVal.tolist()) + "\n")
            f.write("idsSex: " + str(idsSexVal.tolist()) + "\n")
            f.write("idsAge: " + str(idsAgeVal.tolist()) + "\n")
            f.write("idsApoe: " + str(idsApoeVal.tolist()) + "\n")

        print("Done!")

    # convert accs into a dict for all the epochs
    accsProcessed = {}
    for key in accs[0].keys():
        accsProcessed[key] = [acc[key] for acc in accs]


    foldResults.append(accsProcessed)


    # # plot the valLosses
    # valLosses = pd.DataFrame(valLosses)
    # plt.figure(figsize=(10, 6))
    # for col in valLosses.columns:
    #     plt.plot(valLosses[col], label=col)
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation Loss')
    # plt.title('Validation Loss per Task')
    # plt.legend()
    # #plt.savefig(f"valLossesFold{fold}.png")
    # plt.show()

    # clean up memory

    del X_train, X_test
    del y_trainBraak, y_testBraak
    del y_trainCerad, y_testCerad
    del y_trainCogdx, y_testCogdx
    del y_trainSex, y_testSex
    del y_trainAge, y_testAge
    del y_trainApoc, y_testApoc
    del y_trainMTL_sex, y_testMTL_sex
    del y_train_MTLage, y_test_MTLage
    del y_train_MTLapoe, y_test_MTLapoe
    del individual_X_test, individual_X_train
    del train_loaderMTLAge, test_loaderMTLAge
    del train_loaderMTLsex, test_loaderMTLsex
    del train_loaderMTLApoe, test_loaderMTLApoe
    del modelMTLsex, modelMTLAge, modelMTLApoe
    del loss_fnBraak_coral, loss_fnsCerad, loss_fnsCogdx, loss_fnsAge, loss_fnsSex, loss_fnsApoc
    del optimizerBraak, optimizerCerad, optimizerCogdx, optimizerAge, optimizerSex, optimizerApoe
    del y_train_coral_braak, y_test_coral_braak
    gc.collect()

foldProcessed = {}

foldResults = pd.DataFrame(foldResults)


print(foldResults)
# #export this data to a txt file called outputMTL.txt
# with open("outputMTL.txt", "w") as f:
#     for i in range(len(foldbraakacc)):
#         f.write(f"Fold {i + 1}\n")
#         f.write(f"braakacc: {foldbraakacc[i]}\n")
#         f.write(f"braakpm1acc: {foldbraakpm1acc[i]}\n")
#         f.write(f"braakCeradacc: {foldbraakCeradacc[i]}\n")
#         f.write(f"braakCeradpm1acc: {foldbraakCeradpm1acc[i]}\n")
#         f.write(f"braakCogdxacc: {foldbraakCogdxacc[i]}\n")
#         f.write(f"braakCogdxpm1acc: {foldbraakCogdxpm1acc[i]}\n")
#         f.write(f"braakSexacc: {foldbraakSexacc[i]}\n")
#         f.write(f"braakSexpm1acc: {foldbraakSexpm1acc[i]}\n")
#         f.write(f"braakAgeacc: {foldbraakAgeacc[i]}\n")
#         f.write(f"braakAgepm1acc: {foldbraakAgepm1acc[i]}\n")
#         f.write(f"braakApoeacc: {foldbraakApoeacc[i]}\n")
#         f.write(f"braakApoepm1acc: {foldbraakApoepm1acc[i]}\n\n")

for col in foldResults.columns:
    foldResults[col] = foldResults[col].apply(lambda x: pd.Series(x).mean())
# average all the result per epoch for all the folds

print(foldResults)

columns = ["CERAD", "Braak", "Cogdx", "Sex", "Age", "Apoe"]
columnspm1 = ["CERADpm1", "Braakpm1", "Cogdxpm1", "Sexpm1", "Agepm1", "Apoepm1"]

for col in columns:
    plt.plot(foldResults[col], label=col)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracies with 1k genes")
plt.show()

for col in columnspm1:
    plt.plot(foldResults[col], label=col)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracies within 1 stage with 1k genes")
plt.show()

# plot_confusion_matrix(y_testBraak, coral_predict(modelBraak(torch.tensor(X_test, dtype=torch.float32))), classes=[0, 1, 2, 3, 4], title='Confusion Matrix for Braakscore Model')
