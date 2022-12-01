import torch
import torch.nn as nn
from module import conv_block, up_conv, Attn


class Encoder(nn.Module):
    def __init__(self, C_lst=[17, 32, 64, 128, 256]):
        super(Encoder, self).__init__()
        self.enc = nn.ModuleList([conv_block(ch_in=C_lst[0],ch_out=C_lst[1])])
        for ch_in, ch_out in zip(C_lst[1:-1], C_lst[2:]):
            self.enc.append(
                nn.Sequential(*[
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    conv_block(ch_in=ch_in, ch_out=ch_out)
                ])
            )

    def forward(self, x):
        skips = []
        for i in range(0, len(self.enc)):
            x = self.enc[i](x)
            skips.append(x)
        return x, skips[:-1]


class Decoder(nn.Module):
    def __init__(self, C_lst=[512, 256, 128, 64, 32]):
        super(Decoder, self).__init__()
        self.dec = nn.ModuleList([])
        for ch_in, ch_out in zip(C_lst[0:-1], C_lst[1:]):
            self.dec.append(
                nn.ModuleList([
                    up_conv(ch_in=ch_in, ch_out=ch_out),
                    conv_block(ch_in=ch_out * 2, ch_out=ch_out)
                ])
            )

    def forward(self, x, skips):
        skips.reverse()
        for i in range(0, len(self.dec)):
            upsample, conv = self.dec[i]
            x = upsample(x)
            x = conv(torch.cat((x, skips[i]), dim=1))
        return x


class Seq2Map(nn.Module):
    def __init__(self, 
                 input_dim=4,
                 num_hidden=128,
                 dropout=0.1, 
                 device=torch.device('cuda'),
                 max_length=3000,
                 **kwargs):
        super(Seq2Map, self).__init__(**kwargs)
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([num_hidden])).to(device)
        
        self.tok_embedding = nn.Embedding(input_dim, num_hidden)
        self.pos_embedding = nn.Embedding(max_length, num_hidden)
        self.layer = Attn(dim=num_hidden, query_key_dim=num_hidden, dropout=dropout)

    def forward(self, src):
        batch_size, src_len = src.shape[:2]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.tok_embedding(src) * self.scale
        src = self.dropout(src + self.pos_embedding(pos))
        attention = self.layer(src)
        return attention

    
class RFold_Model(nn.Module):
    def __init__(self, args):
        super(RFold_Model, self).__init__()

        c_in, c_out, c_hid = 1, 1, 32
        C_lst_enc = [c_in, 32, 64, 128, 256, 512]
        C_lst_dec = [2*x for x in reversed(C_lst_enc[1:-1])] + [c_hid]

        self.encoder = Encoder(C_lst=C_lst_enc)
        self.decoder = Decoder(C_lst=C_lst_dec)
        self.readout = nn.Conv2d(c_hid, c_out, kernel_size=1, stride=1, padding=0)
        self.seq2map = Seq2Map(input_dim=4, num_hidden=args.num_hidden, dropout=args.dropout)

    def forward(self, seqs):
        attention = self.seq2map(seqs)
        x = (attention * torch.sigmoid(attention)).unsqueeze(0)
        latent, skips = self.encoder(x)
        latent = self.decoder(latent, skips)
        y = self.readout(latent).squeeze(1)
        return torch.transpose(y, -1, -2) * y