import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .ccc import mean_ccc


def evaluate(model, dataset, dataset_name, aggregate="majority", modalities=["ecg"]):
    """
    Performs evaluation of trained supervised models
    Args:
        model (pytorch_lightning.LightningModule): Supervised model to evaluate.
        dataset (torch.utils.data.Dataset): Test dataset.
        dataset_name (str): Name of dataset.
        aggregate (str): Method to aggregate instance-level outputs
    Returns:
        global_metrics (dict): Dictionary containing performance metrics
    """
    model.eval()
    print("Evaluating model...")

    y_true, y_pred, y_name = [], [], []
    for data, label, name in tqdm(dataset):
        # move inputs to device
        if isinstance(data, dict):
            data = {key: data[key].to(model.device) for key in modalities}
        else:
            data = data.to(model.device)

        label = label.to(model.device)
        with torch.no_grad():
            preds, _ = model(data, label)
            # save labels and predictions
            y_true.append(label)
            y_pred.append(preds)
            y_name.extend(name)

    # stack arrays for evaluation
    y_true = torch.cat(y_true, dim=0).squeeze().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).squeeze().cpu().numpy()
    y_name = np.array(y_name)

    if "EPIC" in dataset_name:
        return y_pred, None
