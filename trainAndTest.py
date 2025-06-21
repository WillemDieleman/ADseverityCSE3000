import torch
import numpy as np
import torch.nn.functional as F

def label_to_coral(labels, num_classes):
    # Creates (batch, num_classes - 1) ordinal vectors
    return torch.tensor([[1.0 if i < y else 0.0 for i in range(num_classes - 1)] for y in labels], dtype=torch.float32)


def coral_loss(logits, levels, weight=None):
    # logits: raw output from model (before sigmoid)
    return F.binary_cross_entropy_with_logits(logits, levels, weight=weight)


def coral_predict(probas):
    return (probas > 0.5).sum(dim=1)

def train(dataloader, model, loss_fn, optimizer, regression=False):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        if regression:
            # For regression, use MSE loss
            loss = loss_fn(pred.squeeze(), y)
        else:
            loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # print training loss
    total_loss /= num_batches
    print(f"Avg TRAINING loss: {total_loss:>8f}\n")


def test(dataloader, model, loss_fn, ordinal=True):
    model.eval()
    test_loss, correct, correctplusmin1, correctplusmin2 = 0, 0, 0, 0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    idsPred = dict()
    with torch.no_grad():
        for X, y, ids in dataloader:
            pred = model(X)
            if ordinal:
                # Compute the loss
                test_loss += coral_loss(pred, y).item()
                predicted_class = coral_predict(pred)
                true_class = y.sum(dim=1).long()  # reconstruct class from coral label
                correct += (predicted_class == true_class).float().sum().item()
                diff = (predicted_class - true_class).abs()

                for id, pred_class in zip(ids, predicted_class):
                    id = str(id)  # Ensure ID is a string
                    pred_class = pred_class.item()  # Convert tensor to Python int
                    if id not in idsPred:
                        idsPred[id] = {}
                    if pred_class not in idsPred[id]:
                        idsPred[id][pred_class] = 0
                    idsPred[id][pred_class] += 1  # Increment count for this prediction

                correctplusmin1 += (diff <= 1).float().sum().item()
                correctplusmin2 += (diff <= 2).float().sum().item()
            else:
                test_loss += loss_fn(pred.squeeze(), y).item()
                correct += (pred.argmax(dim=1) == y).float().sum().item()

    test_loss /= num_batches
    correct /= size
    correctplusmin1 /= size
    correctplusmin2 /= size



    print(
        f"Avg loss: {test_loss:>8f}, Correct: {correct * 100:>0.1f}%, Correct within 1 score: {correctplusmin1 * 100:>0.1f}%, Correct within 2 scores: {correctplusmin2 * 100:>0.1f}%\n")



    return test_loss, correct, correctplusmin1, idsPred


def trainMTL(dataloader, model, loss_fns, optimizer, device="cpu"):
    model.train()
    correct = 0
    total_loss = 0
    num_batches = 0
    for batch, (X, y, ids) in enumerate(dataloader):
        num_batches += 1
        if batch > 50:
            break

        X, y = X.to(device), [yi.to(device) for yi in y]  # Move data to device
        pred = model(X)
        loss = 0
        for i in range(len(pred)):
            # CORAL uses float32 ordinal targets

            if model.task_types[i] == "coral" or model.task_types[i] == "classification":
                loss += loss_fns[i](pred[i], y[i])
            elif model.task_types[i] == "regression":
                loss += loss_fns[i](pred[i].squeeze(), y[i])

            if i == 0:
                if model.task_types[i] == "coral":
                    # For the first task, calculate accuracy
                    pred_class = coral_predict(pred[i])
                    true_class = y[i].sum(dim=1).long()
                    correct += (pred_class == true_class).float().sum().item()
                elif model.task_types[i] == "classification":
                    # For the first task, calculate accuracy
                    pred_class = pred[i].argmax(dim=1)
                    true_class = y[i]
                    correct += (pred_class == true_class).float().sum().item()
                # pred_class = (pred[i] > 0.5).sum(dim=1)
                # true_class = y[i].sum(dim=1)
                # correct += (pred_class == true_class).sum().item()
        total_loss += loss.item() / len(pred)

        optimizer.zero_grad()
        loss = loss / len(pred)
        loss.backward()
        optimizer.step()


        # print training loss for the first task
    print(f"Avg TRAINING loss for task 0: {total_loss/num_batches:>8f}")
    print(f"Avg TRAINING accuracy for task 0: {correct / len(dataloader.dataset) * 100:>0.1f}%")

    return total_loss / num_batches


def testMTL(dataloader, model, loss_fns, task_types, device="cpu"):
    model.eval()
    test_loss = 0
    l = len(loss_fns)

    lossArr = np.zeros(l)
    accArr = np.zeros(l)
    accArrpm1 = np.zeros(l)
    accArrpm2 = np.zeros(l)
    idsPred = dict()

    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    with torch.no_grad():
        for X, y, ids in dataloader:
            X, y = X.to(device), [yi.to(device) for yi in y]  # Move data to device
            pred = model(X)
            loss = 0
            for i in range(len(pred)):
                target = y[i]
                output = pred[i]



                if task_types[i] == "coral" or task_types[i] == "classification":
                    temp_loss = loss_fns[i](output, target)
                elif task_types[i] == "regression":
                    temp_loss = loss_fns[i](output.squeeze(), target)
                loss += temp_loss
                lossArr[i] += temp_loss.item()

                if task_types[i] == "coral" or task_types[i] == "classification":
                    if task_types[i] == "coral":
                        pred_class = (output > 0.5).sum(dim=1)
                        true_class = target.sum(dim=1)
                    else:
                        pred_class = output.argmax(dim=1)
                        true_class = target
                    accArr[i] += (pred_class == true_class).sum().item()
                    diff = (pred_class - true_class).abs()
                    accArrpm1[i] += (diff <= 1).sum().item()
                    # --- Store predictions per task per ID ---
                    for id_val, pred_val in zip(ids, pred_class):
                        id_str = str(id_val)
                        pred_int = int(pred_val.item())
                        if id_str not in idsPred:
                            idsPred[id_str] = {}
                        if i not in idsPred[id_str]:
                            idsPred[id_str][i] = {}
                        if pred_int not in idsPred[id_str][i]:
                            idsPred[id_str][i][pred_int] = 0
                        idsPred[id_str][i][pred_int] += 1  # Increment count for this prediction

                elif task_types[i] == "regression":
                    # Optional: You can log MAE or RMSE here
                    pass  # Leave accArr metrics zero for regression

            test_loss += loss.item()


    test_loss /= num_batches
    for i in range(len(lossArr)):
        lossArr[i] /= num_batches
        if task_types[i] == "coral" or task_types[i] == "classification":
            accArr[i] /= size
            accArrpm1[i] /= size
            print(
                f"Task {i} [{task_types[i]}]: Loss={lossArr[i]:.4f}, Acc={accArr[i] * 100:.1f}%, ±1={accArrpm1[i] * 100:.1f}%")
        elif task_types[i] == "regression":
            print(f"Task {i} [Regress]: Loss={lossArr[i]:.4f} (MSE)")

    print(f"Avg loss: {test_loss:>8f}\n")
    return test_loss, accArr.mean(), lossArr[0], accArr, accArrpm1, idsPred

def predictIds(ids, dicts, task_types):
    acc = dict()
    accOneAway = dict()
    for id in ids:

        for task in range(len(dicts)):
            if task_types[task] == "regression":
                continue
            if task >= 3:
                continue
            if task not in acc.keys():
                acc[task] = 0
                accOneAway[task] = 0
            y = dicts[task][id]
            total = 0
            for j in range(4):
                if j in ids[id][task].keys():
                    total += ids[id][task][j]

            max = 0
            indexMax = 0
            for j in range(4):
                if j not in ids[id][task].keys():
                    continue
                score = ids[id][task][j]
                if score > max:
                    max = score
                    indexMax = j

            if y == indexMax:
                acc[task] += 1
            if np.abs(y - indexMax) <= 1:
                accOneAway[task] += 1
    for task in acc.keys():
        acc[task] = acc[task] / len(ids)
        accOneAway[task] = accOneAway[task] / len(ids)
    # acc = acc / len(ids)
    # accOneAway = oneAway / len(ids)
    return acc, accOneAway