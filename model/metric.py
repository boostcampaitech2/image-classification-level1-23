import torch


def accuracy(output, target):
    with torch.no_grad():
        # pred = torch.argmax(output, dim=1)
        pred = torch.max(output.data, dim=1)
        # assert pred.shape[0] == len(target)
        correct = 0
        # correct += torch.sum(pred == target).item()
        correct += (pred.indices == target).sum().item()
    return correct / len(target)

def f1_score(output, target, is_training=False):
    """
        2 * (precision * recall) / (precision + recall)
        
        ㄴ precision() : 정미도
        ㄴ recall : 재현률
    """
    with torch.no_grad():
        assert output.shape[0] == len(target)
        # correct = 0
        # correct = torch.argmax(output, dim=1)
        # tp = (target * correct).sum()
        # fp = ((18 - target) * correct).sum()
        # fn = (target * (18 - correct)).sum()        
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn )
        if target.ndim == 2:
            target = target.argmax(dim=1)
        # pred = torch.argmax(output, dim=1)
        pred = torch.max(output.data, dim=1)
        pred = pred.indices
        tp = (target * pred).sum().to(torch.float32)
        tn = ((1 - target) * (1 - pred)).sum().to(torch.float32)
        fp = ((1 - target) * pred).sum().to(torch.float32)
        fn = (target * (1 - pred)).sum().to(torch.float32)
        
        epsilon = 1e-7
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2* (precision*recall) / (precision + recall)
        f1.requires_grad = False    
    return f1

def top_10_acc(output, target, k=10):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
