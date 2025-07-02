from typing import Callable, Dict, Tuple
from logger import create_run
import torch
import adaptations
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import utils
import warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA IS AVAILABLE: {torch.cuda.is_available()}")

def train(network, train_data, config, run, epoch: int, nc: Dict[Tuple, float], c: Dict[Tuple, float], spacing: Callable = adaptations.base, targets="dynamic"):
    """Trains network according using specified target spacing method, returns new spaced values for next epoch"""

    optimizer = optim.SGD(network.parameters(), lr=config.learning_rate, momentum=config.momentum)
    network.train()
    
    outputs = {}
    class_targets: Dict[Tuple, float] = c
    nclass_targets: Dict[Tuple, float] = nc

    for batch_idx, (data, target) in enumerate(train_data):
        data = data.to(device)
        # new_target = generate_target(target)
        if targets != "dynamic":
            new_target = utils.generate_alt_target(target, targets).to(device)
        else:
            new_target = utils.apply_class_values(utils.generate_target(target), nclass_targets, class_targets).to(device)
        optimizer.zero_grad()
        output = network(data)
        outputs = utils.add_outputs_by_class(outputs, output, target)
        
        loss = utils.one_hot_nll_loss(output, new_target)
        run.log({"loss":loss, "epoch":epoch + batch_idx / len(train_data)})

        loss.backward()
        optimizer.step()

        if batch_idx % config.log_interval == 0:
            print(f"Epoch: {epoch} [{batch_idx*len(data)}/{len(train_data)*config.batch_size_train} Loss: {loss.item()}]", end="\r")

            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

    return spacing(class_targets, nclass_targets, outputs, config.push_multiplier)


def test(network, test_data, config, run, nc, c, epoch=0.0, targets="dynamic"):
    """Tests network using test dataset"""
    network.eval()

    oh_classes = list(c.keys())
    class_target_vectors = torch.tensor(oh_classes, dtype=torch.float32)
    if targets == "dynamic":
        class_target_vectors = utils.apply_class_values(class_target_vectors, nc, c)
    elif targets == "soft":
        class_target_vectors[class_target_vectors == 1] = 0.8
        class_target_vectors[class_target_vectors == 1] = 0.2

    for class_vector in c:
        num = class_vector.index(1)
        run.log({
            f"class value ({num})": c[class_vector],
            f"non-class value ({num})": nc[class_vector],
            "epoch": epoch
        })

    for class_tuple in c.keys():
        separation = abs(c[class_tuple] - nc[class_tuple])
        run.log({f"class_nc_separation_{class_tuple.index(1)}": separation, "epoch": epoch})

    separations = [abs(c[k] - nc[k]) for k in c.keys()]
    run.log({
        "avg_separation": np.mean(separations),
        "min_separation": np.min(separations), 
        "max_separation": np.max(separations),
        "epoch": epoch
    })

    adaptations.set_epoch(epoch)

    test_loss = 0
    total_confidence = 0.0
    total_soft_accuracy = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_data:
            data = data.to(device)
            if targets != "dynamic":
                target_vecs = utils.generate_alt_target(target, targets).to(device)
            else:
                target_vecs = utils.generate_target(target).to(device)

            output = network(data)
            probs = output.exp()
            
            if targets != "dynamic":
                loss = utils.one_hot_nll_loss(output, target_vecs)
            else:
                loss = utils.one_hot_nll_loss(output, utils.apply_class_values(target_vecs, nc, c))
            test_loss += loss * data.size(0)

            ## Calculate cosine sim of closest class_target to output
            probs_norm = F.normalize(probs, dim=1)
            class_targets_norm = F.normalize(class_target_vectors, dim=1)

            # Cosine similarities: [B, N]
            similarities = torch.matmul(probs_norm, class_targets_norm.T)

            # Max similarity per row
            max_similarities, _ = similarities.max(dim=1)

            # Add to your total_confidence
            total_confidence += max_similarities.sum()

            ## Predicted class (based on nearest target vector)
            if targets == "dynamic":
                # print(probs)
                predicted = utils.predict_by_nearest_target(probs, c, nc)
                total_correct += predicted.eq(target).sum().item()
            else:
                predicted = output.argmax(dim=1)
                total_correct += predicted.eq(target).sum().item()

            total_samples += data.size(0)

    test_loss /= total_samples
    avg_soft_acc = total_soft_accuracy / total_samples
    avg_conf = total_confidence / total_samples
    hard_acc = total_correct / total_samples
    log_data = {
        "hard accuracy": hard_acc,
        "confidence": avg_conf,
        "test loss": test_loss,
        "avg cosine similarity (confidence)": avg_conf,
        "epoch": epoch,
    }

    run.log(log_data)

def run_experiment(model, train_data, test_data, config):
    run = create_run(config)
    adaptations.set_run(run)
    adaptation: Callable = adaptations.ADAPTATION_REGISTRY[config.adaptation]

    model = model.to(device)
    
    pre_train: Callable = warmup.PRETRAINING_REGISTRY[config.warmup]

    nc, c = pre_train(
        config=config,
        network=model,
        test_data=test_data,
        device=device,
        spacing=adaptation,
        nudge=config.nudge
    )
    test(model, test_data, config, run, nc=nc, c=c, epoch=1, targets=config.targets)
    for epoch in range(1, config.epochs + 1):
        nc, c = train(model, train_data, config, run, nc=nc, c=c, epoch=epoch, spacing=adaptation, targets=config.targets)
        test(model, test_data, config, run, epoch=float(epoch)+0.999, targets=config.targets, c=c, nc=nc)

    run.finish()
