import torch
import torch.nn as nn

class NeuralNetwork3Layer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CoralOrdinalNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # Output layer with (num_classes - 1) ordinal logits
        self.output = nn.Linear(32, num_classes - 1)
        #self.output = nn.Linear(16, num_classes - 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.output(x)
        return logits

#multi task model
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_list, task_types, input_dim=1000):
        super(MultiTaskModel, self).__init__()
        self.task_types = task_types
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(input_dim, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.task_layers = nn.ModuleList()

        for out_dim, task_type in zip(num_classes_list, task_types):
            if task_type == "coral":
                final = out_dim - 1
            elif task_type == "regression":
                final = 1
            elif task_type == "classification":
                final = out_dim
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            layer = nn.Sequential(
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                # nn.Linear(128, 64),
                # nn.BatchNorm1d(64),
                # nn.ReLU(),
                nn.Linear(64, final)
            )

            self.task_layers.append(layer)

    def forward(self, x):
        shared_representation = self.shared_layer(x)
        task_outputs = []
        for layer, task_type in zip(self.task_layers, self.task_types):
            out = layer(shared_representation)
            # if task_type == "coral":
            #     out = nn.sigmoid(x)
            # For regression, leave it raw
            task_outputs.append(out)
        return task_outputs

    # def forward(self, x):
    #     x = self.feature_extractor(x)
    #     logits = self.output(x)
    #     return logits

class EmbeddingsModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
