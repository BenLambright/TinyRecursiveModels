import torch
import torch.distributed as dist

class SumMatchEvaluator:
    # Tell evaluate() what it must request/keep around
    required_outputs = {"preds", "answers"}   # adjust keys to match your batch/preds dicts

    def begin_eval(self):
        self.correct = 0
        self.total = 0

    def update_batch(self, batch, preds):
        # grab CPU tensors
        gold = batch["answers"].detach().cpu()
        pred = preds["preds"].detach().cpu()

        # flatten per example, sum
        gold_sum = gold.view(gold.size(0), -1).sum(dim=1)
        pred_sum = pred.view(pred.size(0), -1).sum(dim=1)

        self.correct += (gold_sum == pred_sum).sum().item()
        self.total += gold.size(0)

    def result(self, save_path, rank, world_size, group=None):
        # reduce to rank 0
        device = torch.device("cpu")
        t = torch.tensor([self.correct, self.total], dtype=torch.long, device=device)
        if world_size > 1:
            dist.reduce(t, dst=0, op=dist.ReduceOp.SUM, group=group)

        if rank != 0:
            return None

        correct, total = t.tolist()
        acc = correct / max(total, 1)
        return {"eval/sum_match_acc": acc}
