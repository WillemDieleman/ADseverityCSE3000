This repository is part of the CSE3000 Bachelor Thesis of Computer Science and Engineering at TU Delft. The paper can be found at the following link: https://repository.tudelft.nl/record/uuid:2e29eb85-fa8a-4e7b-8598-ef9838931b01.

# A Multi Task Learning approach to classifying the severity of Alzheimer’s disease using single-cell gene expression data

This repository contains the code for the paper "A Multi Task Learning approach to classifying the severity of Alzheimer’s disease using single-cell gene expression data".

This research used the ROMSAP dataset, which can be downloaded from the following link: https://www.synapse.org/Synapse:syn3219045. A request for access is needed, which can be done here: https://adknowledgeportal.synapse.org/Data%20Access.

This repository is organized as follows:

Main.py contains the main code for training and evaluating the model. In there, you can select which tasks should be included in the MTL model. The available tasks are: Braak, CERAD, Cogdx, Age, Sex, APOE, Cell Type. Make sure to also change the amount of output classes and dicts to the correct task order. 

All preprocessing is done in the dataset.py file. Definitions of the models can be found in models.py. Definitions of the train and test functions can be found in trainAndTest.py.

The folder oldModels contains the old models that were not used in the paper, but were created during the development of this project.

In the folder analysis you can find all the analysis scripts that were used to create the figures and tables in the paper.
