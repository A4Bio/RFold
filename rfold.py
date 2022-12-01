import time
import torch
import numpy as np
from tqdm import tqdm
from utils import cuda
from API import evaluate_result
from model import RFold_Model


# predefine a base_matrix
_max_length = 1005
base_matrix = torch.ones(_max_length, _max_length)
for i in range(_max_length):
    st, en = max(i-3, 0), min(i+3, _max_length-1)
    for j in range(st, en + 1):
        base_matrix[i, j] = 0.

def constraint_matrix(x):
    base_a, base_u, base_c, base_g = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return (au_ua + cg_gc + ug_gu) * base_matrix[:length, :length].to(x.device)

def row_col_softmax(y):
    row_softmax = torch.softmax(y, dim=-1)
    col_softmax = torch.softmax(y, dim=-2)
    return 0.5 * (row_softmax + col_softmax)

def row_col_argmax(y):
    y_pred = row_col_softmax(y)
    y_hat = y_pred + torch.randn_like(y) * 1e-12
    col_max = torch.argmax(y_hat, 1)
    col_one = torch.zeros_like(y_hat).scatter(1, col_max.unsqueeze(1), 1.0)
    row_max = torch.argmax(y_hat, 2)
    row_one = torch.zeros_like(y_hat).scatter(2, row_max.unsqueeze(2), 1.0)
    int_one = row_one * col_one 
    return int_one


class RFold(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.config = args.__dict__

        self.model = self._build_model()
        self.criterion = torch.nn.MSELoss()

    def _build_model(self, **kwargs):
        return RFold_Model(self.args).to(self.device)

    def test_one_epoch(self, test_loader, **kwargs):
        # note that the model is under the training mode for bn/dropout
        self.model.train()
        eval_results, run_time = [], []
        test_pbar = tqdm(test_loader)
        for batch in test_pbar:
            contacts, seq_lens, seq_ori = batch
            contacts, seq_ori = cuda(
                (contacts.float(), seq_ori.float()), device=self.device)

            # predict
            seqs = torch.argmax(seq_ori, axis=-1)
            s_time = time.time()
            with torch.no_grad():
                pred_contacts = self.model(seqs)

            pred_contacts = row_col_argmax(pred_contacts) * constraint_matrix(seq_ori)

            # interval time
            interval_t = time.time() - s_time
            run_time.append(interval_t)

            eval_result = list(map(lambda i: evaluate_result(pred_contacts.cpu()[i],
                                                                     contacts.cpu()[i]), range(contacts.shape[0])))
            eval_results += eval_result

        p, r, f1 = zip(*eval_results)
        return np.average(f1), np.average(p), np.average(r), np.average(run_time)