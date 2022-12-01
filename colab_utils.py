import torch
import os.path as osp
import numpy as np
import torch.nn.functional as F


seq_dict = {
    'A': 0,
    'U': 1,
    'C': 2,
    'G': 3
}


def base_matrix(_max_length, device):
    base_matrix = torch.ones(_max_length, _max_length)
    for i in range(_max_length):
        st, en = max(i-3, 0), min(i+3, _max_length-1)
        for j in range(st, en + 1):
            base_matrix[i, j] = 0.
    return base_matrix.to(device)

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
    return (au_ua + cg_gc + ug_gu) * base_matrix(x.shape[1], x.device)

def sequence2onehot(seq, device):
    seqs = list(map(lambda x: seq_dict[x], seq))
    return torch.tensor(seqs).unsqueeze(0).to(device)

def get_cut_len(l):
    return (((l - 1) // 16) + 1) * 16

def process_seqs(seq, device):
    seq_len = len(seq)
    seq = sequence2onehot(seq, device=device)
    nseq_len = get_cut_len(seq_len)
    nseq = F.pad(seq, (0, nseq_len - seq_len))
    nseq_one_hot = F.one_hot(nseq).float()
    return nseq, nseq_one_hot, seq_len
  
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

def ct_file_output(pairs, seq, seq_name, save_result_path):
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]-1] = int(I[1])
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    np.savetxt(osp.join(save_result_path, seq_name.replace('/','_'))+'.ct', (temp), delimiter='\t', fmt="%s", header='>seq length: ' + str(len(seq)) + '\t seq name: ' + seq_name.replace('/','_') , comments='')
    return

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def save_ct(predict_matrix, seq_ori, name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    dot_list = seq2dot((seq_tmp+1).squeeze())
    letter = 'AUCG'
    seq_letter = ''.join([letter[item] for item in np.nonzero(seq_ori)[:,1]])
    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    cur_pred = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    ct_file_output(cur_pred, seq_letter, name, './')
    return 

def visual_get_bases(seq):
    a_bases, u_bases, c_bases, g_bases = [], [], [], []
    for ii, s in enumerate(seq):
        if s == 'A': a_bases.append(ii+1)
        if s == 'U': u_bases.append(ii+1)
        if s == 'C': c_bases.append(ii+1)
        if s == 'G': g_bases.append(ii+1)
    a_bases = ''.join([str(s)+',' for s in a_bases])[:-1]
    u_bases = ''.join([str(s)+',' for s in u_bases])[:-1]
    c_bases = ''.join([str(s)+',' for s in c_bases])[:-1]
    g_bases = ''.join([str(s)+',' for s in g_bases])[:-1]
    return a_bases, u_bases, c_bases, g_bases