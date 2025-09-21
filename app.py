import json, re, torch
import torch.nn as nn
import streamlit as st
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(".")
BPE_DIR = ROOT/"data/processed/bpe"
CKPT    = ROOT/"checkpoints/best.pt"

PAD, BOS, EOS, UNK = 0,1,2,3
SPACE = "▁"
SPECIALS = {"<pad>", "<bos>", "<eos>", "<unk>"}

# ---------- BPE utils ----------
def _merge_seq(seq, pair):
    a,b = pair; out=[]; i=0
    while i < len(seq):
        if i+1 < len(seq) and seq[i]==a and seq[i+1]==b:
            out.append(a+b); i+=2
        else:
            out.append(seq[i]); i+=1
    return out

def load_bpe(path):
    with open(path, encoding="utf-8") as f: return json.load(f)

def apply_bpe(line, model, add_bos_eos=True):
    merges=model["merges"]; itos=model["itos"]; stoi={t:i for i,t in enumerate(itos)}
    line=re.sub(r"\s+"," ", line.strip().lower())
    pieces=[]
    if line:
        for w in line.split(" "):
            seq=[SPACE]+list(w)
            for a,b in merges: seq=_merge_seq(seq,(a,b))
            pieces.extend(seq)
    ids=[stoi.get(p,UNK) for p in pieces]
    return [BOS]+ids+[EOS] if add_bos_eos else ids

def ids_to_text(ids, itos):
    toks=[itos[i] if 0<=i<len(itos) else "<unk>" for i in ids]
    toks=[t for t in toks if t not in SPECIALS]
    s=""
    for t in toks:
        s+=(" "+t[1:] if t.startswith(SPACE) else t)
    return s.strip()

# ---------- Model ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb=256, hid=512, layers=2, dropout=0.3):
        super().__init__()
        self.emb=nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.rnn=nn.LSTM(emb, hid//2, num_layers=layers, batch_first=True,
                         dropout=dropout if layers>1 else 0.0, bidirectional=True)
    def forward(self,x):
        emb=self.emb(x); out,(h,c)=self.rnn(emb); return out,(h,c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb=256, hid=512, layers=4, dropout=0.3):
        super().__init__()
        self.emb=nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.rnn=nn.LSTM(emb, hid, num_layers=layers, batch_first=True,
                         dropout=dropout if layers>1 else 0.0)
        self.fc=nn.Linear(hid,vocab_size)
    def forward(self,y,state):
        emb=self.emb(y); out,state=self.rnn(emb,state); return self.fc(out),state

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb=256, hid=512, enc_layers=2, dec_layers=4, dropout=0.3):
        super().__init__()
        self.encoder=Encoder(src_vocab,emb,hid,enc_layers,dropout)
        self.decoder=Decoder(tgt_vocab,emb,hid,dec_layers,dropout)
        self.bridge_h=nn.Linear(hid,hid)
        self.bridge_c=nn.Linear(hid,hid)
    def init_dec(self,h,c):
        H=torch.cat([h[-2],h[-1]],dim=-1); C=torch.cat([c[-2],c[-1]],dim=-1)
        dh=self.bridge_h(H).unsqueeze(0).repeat(self.decoder.rnn.num_layers,1,1)
        dc=self.bridge_c(C).unsqueeze(0).repeat(self.decoder.rnn.num_layers,1,1)
        return dh,dc
    @torch.no_grad()
    def greedy(self,src_ids,max_len=128,device="cpu"):
        src=torch.tensor([src_ids],dtype=torch.long,device=device)
        enc_out,(h,c)=self.encoder(src)
        dh,dc=self.init_dec(h,c)
        y=torch.tensor([[BOS]],dtype=torch.long,device=device)
        out=[]
        for _ in range(max_len):
            logits,(dh,dc)=self.decoder(y,(dh,dc))
            nid=logits[:,-1,:].argmax(dim=-1).item()
            if nid==EOS: break
            out.append(nid)
            y=torch.cat([y,torch.tensor([[nid]],device=device)],dim=1)
        return out

# ---------- Load artifacts ----------
device="cuda" if torch.cuda.is_available() else "cpu"
src_bpe=load_bpe(BPE_DIR/"bpe_src.json")
tgt_bpe=load_bpe(BPE_DIR/"bpe_tgt.json")
src_vocab=len(src_bpe["itos"]); tgt_vocab=len(tgt_bpe["itos"])
tgt_itos=tgt_bpe["itos"]

model=Seq2Seq(src_vocab,tgt_vocab).to(device)
state=torch.load(CKPT,map_location=device)
model.load_state_dict(state["model"])
model.eval()

# ---------- UI ----------
st.set_page_config(page_title="Urdu → Roman Urdu", page_icon="📝")
st.title("Urdu → Roman Urdu Translator")
st.caption("BiLSTM Encoder–Decoder with BPE (from scratch) · PyTorch")

inp=st.text_area("Urdu input:", height=120, value="میں تم سے محبت کرتا ہوں")
max_len=st.slider("Max output length",32,256,128,step=16)
if st.button("Translate"):
    if not inp.strip():
        st.warning("Please enter Urdu text.")
    else:
        src_ids=apply_bpe(inp,src_bpe,add_bos_eos=True)
        pred_ids=model.greedy(src_ids,max_len=max_len,device=device)
        out=ids_to_text(pred_ids,tgt_itos)
        st.subheader("Roman Urdu")
        st.write(out if out else "(empty)")
