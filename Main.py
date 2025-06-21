from random import random

from models import *
from dataset import *
from trainAndTest import *

import gc
import numpy as np
import torch
import scanpy as sc
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from datetime import datetime


class Model:
    def __init__(self, task_names=None, task_types=None, input_dim=1000, output_dims=None, epochs=15, lr=0.001, wd=0.001,
                 inputData=None, metadata=None, outputFolder=None, dicts=None, random_state=42, cellTypes=None, device="cpu"):
        self.task_names = task_names
        self.main_task = task_names[0] if task_names else None
        self.task_types = task_types
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.inputData = inputData
        self.metadata = metadata
        self.outputFolder = outputFolder
        self.dicts = dicts
        self.mainDict = dicts[0] if dicts else None
        self.random_state = random_state
        self.individuals = np.array(self.inputData.obs["individualID"].unique())
        self.cellTypes = cellTypes
        self.device = device
        self.train_ids = None
        self.val_ids = None
        self.test_ids = None

        self.model = None
        self.loss_fns = None
        self.optimizer = None
        self.kf = None
        self.foldCount = None
        self.indices = None

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.train_individuals = None
        self.val_individuals = None
        self.test_individuals = None

        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None
    def cleanup(self):
        self.val_ids = None
        self.test_ids = None

        self.model = None
        self.loss_fns = None
        self.optimizer = None
        self.kf = None

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.train_individuals = None
        self.val_individuals = None
        self.test_individuals = None

        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None

    def trainTestsplit(self):
        scores = []
        for dict, name in zip(self.dicts, self.task_names):
            taskSores = np.array([dict[ind] for ind in self.individuals])
            #if name == "Braak" or name == "CERAD" or name == "Cogdx":
            print(name, "score distribution:")
            print(pd.Series(taskSores).value_counts())
            scores.append(taskSores)
        y_tuple_list = list(zip(*scores))
        composite_labels = [str(tup) for tup in y_tuple_list]
        #np.array([self.mainDict[ind] for ind in self.individuals])
        if composite_labels == []:
            composite_labels = np.zeros(len(self.individuals), dtype=str)


        self.kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        foldResults = []
        foldTestResults = []
        fold = 1
        for train_ids, test_ids in self.kf.split(self.individuals, composite_labels):
            self.foldCount = fold
            # if fold != 1:
            #     break
            print(f"Fold {fold}")
            self.train_ids = self.individuals[train_ids]
            self.test_ids = self.individuals[test_ids]
            print("Amount of individuals in train set:", len(self.train_ids))
            print("Amount of individuals in test set:", len(self.test_ids))

            self.featureSelection()
            self.loadData()
            self.createModel()
            accs, testacc = self.runModel()
            foldResults.append(accs)
            foldTestResults.append(testacc)
            fold += 1
            self.cleanup()
            gc.collect()

        avgfold = []
        avgTestfold = []
        for task in foldResults[0].keys():
            taskAcc = []
            taskTestAcc = []
            for i in range(5):
                taskAcc.append(foldResults[i][task])
                taskTestAcc.append(foldTestResults[i][task])
            avgfold.append(np.mean(taskAcc))
            avgTestfold.append(np.mean(taskTestAcc))

        # avgFold = avgfold
        # avgTestFold = avgTestfold
        return avgfold, avgTestfold

        # self.train_ids, self.test_ids = train_test_split(self.individuals, test_size=0.2, stratify=main_scores,
        #                                                      random_state=self.random_state)
        #
        # print("Amount on individuals in train set:", len(self.train_ids))
        # print("Amount on individuals in test set:", len(self.test_ids))

        #print(self.train_ids)

    def featureSelection(self):
        self.X_train = self.inputData[self.inputData.obs["individualID"].isin(self.train_ids)]
        self.X_test = self.inputData[self.inputData.obs["individualID"].isin(self.test_ids)]
        self.X_val = self.inputData[self.inputData.obs["individualID"].isin(self.test_ids)] # TODO No cross validation

        print("Train set shape:", self.X_train.shape)
        print("Test set shape:", self.X_test.shape)

        y_train = []
        y_test = []

        for i in range(len(self.task_names)):
            if self.task_names[i] == "Cell Type":
                y_task_train = self.X_train.obs["state"]
                y_task_test = self.X_test.obs["state"]
            else :
                y_task_train = self.X_train.obs["individualID"].map(self.dicts[i])
                y_task_test = self.X_test.obs["individualID"].map(self.dicts[i])
            y_train.append(y_task_train)
            y_test.append(y_task_test)

        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_test  # TODO No cross validation, so use test set as validation set

        self.train_individuals = self.X_train.obs["individualID"]
        self.test_individuals = self.X_test.obs["individualID"]
        self.val_individuals = self.X_val.obs["individualID"]
        selected_indices = select_top_features_multi_task(self.X_train.X, self.y_train, top_k=self.input_dim, agg_method="sum")
        self.indices = selected_indices
        self.X_train = self.X_train[:, selected_indices].X
        self.X_test = self.X_test[:, selected_indices].X
        self.X_val = self.X_val[:, selected_indices].X

        #apply PCA

        # pca = PCA(n_components=self.input_dim)
        # self.X_train = pca.fit_transform(self.X_train)
        # self.X_test = pca.transform(self.X_test)
        # self.X_val = pca.transform(self.X_val)
        # print("Shape of train set after feature selection:", self.X_train.shape)


        for i in range(len(self.task_types)):
            type = self.task_types[i]
            if type == "coral":
                y_train_coral = label_to_coral(self.y_train[i], self.output_dims[i])
                y_test_coral = label_to_coral(self.y_test[i], self.output_dims[i])
                y_val_coral = label_to_coral(self.y_val[i], self.output_dims[i])
                self.y_train[i] = y_train_coral
                self.y_test[i] = y_test_coral
                self.y_val[i] = y_val_coral

    def loadData(self):
        f = torch.float32
        l = torch.long
        types = []
        for task_type in self.task_types:
            if task_type == "classification":
                types.append(l)
            else:
                types.append(f)

        self.trainLoader, self.valLoader, self.testLoader = dataLoader(
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
            types=types,
            test_ids=self.train_individuals,
            train_ids=self.train_individuals,
            val_ids=self.val_individuals
        )

    def createModel(self):
        self.model = MultiTaskModel(
            input_dim=self.input_dim,
            num_classes_list=self.output_dims,
            task_types=self.task_types
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        loss_fns = []
        for i in range(len(self.task_types)):
            type = self.task_types[i]
            if type == "coral":
                loss_fns.append(nn.BCEWithLogitsLoss())
            elif type == "regression":
                loss_fns.append(nn.MSELoss())
            elif type == "classification":
                loss_fns.append(nn.CrossEntropyLoss())

        self.loss_fns = loss_fns

    def runModel(self):
        # self.trainTestsplit()
        # self.featureSelection()
        # self.loadData()
        # self.createModel()

        accs = []
        testAcc = []

        prev_loss = 1e10
        patience = 0

        losses = []
        testLosses = []

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            loss = trainMTL(self.trainLoader, self.model, self.loss_fns, self.optimizer, device=self.device)
            _, _, test_loss, acc, _, ids = testMTL(self.valLoader, self.model, self.loss_fns, self.task_types, device=self.device)

            losses.append(loss)
            testLosses.append(test_loss)
            if self.main_task != "Cell Type":
                val_acc, val_pm1_acc = predictIds(ids, self.dicts, self.task_types)
                for key in val_acc.keys():
                    print(f"{self.task_names[key]} Validation Accuracy: {val_acc[key] * 100:>0.1f}%, One away: {val_pm1_acc[key] * 100:>0.1f}%")
                #print(f"{self.main_task} Validation Accuracy: {val_acc * 100:>0.1f}%, One away: {val_pm1_acc * 100:>0.1f}%")

                accs = val_acc
                testAcc = acc

            if loss > prev_loss:
                patience += 1
            else:
                patience = 0
                prev_loss = loss

            if patience > 3:
                print("Early stopping")
                break
        # export the model to the output folder
        # if self.outputFolder:
        #     torch.save(self.model.state_dict(), f"{self.outputFolder}/models/{self.foldCount}_CTmodel.pth")
        #     print(f"Model saved to {self.outputFolder}/models/{self.foldCount}_CTmodel.pth")
        #     #export the indices and train ids to a txt file in the data folder with the foldnumber as name
        #     np.savetxt(f"{self.outputFolder}/data/{self.foldCount}_CTindices.txt", self.indices, fmt='%d')
        #     np.savetxt(f"{self.outputFolder}/data/{self.foldCount}_CTtestIds.txt", self.test_ids, fmt='%s')
        return accs, testAcc

start_time = datetime.now()

astrocytes = sc.read_h5ad("../data/rosmap/microglia.h5ad")

metadata = pd.read_csv("../data/rosmap/rosmap_clinical.csv")

# astrocytes = sc.read_h5ad("/tudelft.net/staff-umbrella/bachelorAD/data/ROSMAP/inhibitory.h5ad")
# metadata = pd.read_csv("/tudelft.net/staff-umbrella/bachelorAD/data/ROSMAP/ROSMAP_clinical.csv")

scores, _, annListUpdated = preprocess([astrocytes], metadata)
del astrocytes
braakDict = scores[0]
ceradDict = scores[1]
cogdxDict = scores[2]
sexDict = scores[3]
ageDict = scores[4]
apoeDict = scores[5]
ADDict = scores[6]

#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")


test = Model(
    task_names=["Cogdx", "Cerad"],
    task_types=["coral", "coral"],
    input_dim=1000,
    output_dims=[3, 4],
    epochs=100,
    inputData=annListUpdated[0],
    metadata=metadata,
    outputFolder="../allTasks",
    dicts=[cogdxDict, ceradDict],
    random_state=42,
    device=device
)

finalaccs = []
finalTestaccs = []
for i in range(4):
    print(f"Run {i+1}")
    accs, testacc = test.trainTestsplit()
    finalaccs.append(accs)
    finalTestaccs.append(testacc)



finalaccs = np.array(finalaccs)
print(finalaccs)


finalMeans = np.mean(finalaccs, axis=0)
finalStds = np.std(finalaccs, axis=0)
finalTestaccs = np.array(finalTestaccs)

print(finalTestaccs)
finalTestMeans = np.mean(finalTestaccs, axis=0)
finalTestStds = np.std(finalTestaccs, axis=0)

print("Final accuracies:")
print("Individual basis")
for i in range(len(test.task_names)):
    t = test.task_names[i]
    if t == "Braak" or t == "CERAD" or t == "Cogdx" or t == "AD":
        print(f"{test.task_names[i]}: Mean: {finalMeans[i]:.4f} +/- {finalStds[i]:.4f}")
print("Cell basis")
for i in range(len(test.task_names)):
    t = test.task_names[i]
    if t == "Braak" or t == "CERAD" or t == "Cogdx" or t == "AD":
        print(f"{test.task_names[i]}: Mean: {finalTestMeans[i]:.4f} +/- {finalTestStds[i]:.4f}")

end = datetime.now()
print(f"Took {end-start_time} to complete.")
# arr = []
#
# for i in range(6):
#     print(f"Run {i+1}")
#     accs = test.runModel()
#     arr.append(accs)
#     print(arr)

# #average the accs for each epoch
# print(arr)