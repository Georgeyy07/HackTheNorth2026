"""Separate focal class balance and averaging for each labeled dataset."""
import torch
from torch.nn import functional as F
from road_training.train import focal_from_cross_entropy


def dataset_joint_loss(output, targets, datasets, alphas, *, roughness_weight=1.,
                       dataset_weights=None, gamma=2.):
    """Average each dataset on its known labels, then average task domains.

    Kaggle and RoadSens have different annotation priors. Their TRAIN-fitted
    class weights never get pooled. Unknown labels have no gradient.
    """
    if len(datasets) != len(output["roughness"]):
        raise ValueError("Expected one dataset name per window")
    device = output["roughness"].device
    rv = targets["roughness_valid"] & output["patch_valid"]
    dv = targets["disturbance_valid"] & output["patch_valid"]
    rloss = output["roughness"].new_zeros((), dtype=torch.float32)
    if rv.any():
        rloss = F.smooth_l1_loss(output["roughness"].float()[rv], targets["roughness"][rv], beta=1.)
    weights = dataset_weights or {"kaggle":1., "roadsens":1.}
    terms, values = [], {}
    for name, weight in weights.items():
        rows = torch.tensor([n == name for n in datasets], device=device)[:,None]
        valid = dv & rows
        if valid.any():
            y = targets["disturbance"][valid]
            ce = F.binary_cross_entropy_with_logits(output["disturbance_logit"].float()[valid], y.float(), reduction="none")
            loss = (focal_from_cross_entropy(ce, gamma)*alphas[name][y]).mean()
            values[name] = loss
            terms.append(weight*loss)
    dloss = sum(terms, output["roughness"].new_zeros((),dtype=torch.float32))/sum(weights.values())
    return dict(total=roughness_weight*rloss+dloss, roughness=rloss, disturbance=dloss,
                roughness_valid=rv, disturbance_valid=dv, by_dataset=values)
