"""
all_in_one_project.py
Fully working single-file implementation of:
"Advanced Time Series Forecasting with LSTM + Bahdanau Attention"

This version contains:
✔ No syntax errors
✔ Fully fixed baseline
✔ All training + evaluation
✔ report.txt auto generated
✔ Works on CPU (Windows safe)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# -------------------------------------------------------
# Utility
# -------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------------------------------------------------------
# Generate synthetic data
# -------------------------------------------------------
def generate_multivariate_ts(length=4000, seed=42):
    np.random.seed(seed)
    t = np.arange(length)
    df = pd.DataFrame({'t': t})
    df['series1'] = 0.05*t + 3*np.sin(2*np.pi*t/365) + np.random.normal(0,0.5,length)
    df['series2'] = 1.5*np.sin(2*np.pi*t/7) + 0.5*np.cos(2*np.pi*t/30) + np.random.normal(0,0.3,length)
    df['series3'] = 0.02*t + np.random.normal(0,0.2,length)
    spikes = np.random.choice(length, size=int(length*0.01), replace=False)
    df.loc[spikes,'series3'] += np.random.normal(3,1,len(spikes))
    return df[['series1','series2','series3']]

def create_windows(values, seq_len=60, pred_h=12):
    X=[];Y=[]
    for i in range(len(values)-seq_len-pred_h+1):
        X.append(values[i:i+seq_len])
        Y.append(values[i+seq_len:i+seq_len+pred_h, 0])
    return np.array(X), np.array(Y)

def get_splits(seq_len=60, pred_h=12):
    df = generate_multivariate_ts()
    X, Y = create_windows(df.values.astype(np.float32), seq_len, pred_h)
    N=len(X)
    test=int(N*0.15)
    val=int(N*0.10)
    train=N-test-val
    return X[:train],Y[:train], X[train:train+val],Y[train:train+val], X[train+val:],Y[train+val:]

# -------------------------------------------------------
# Scaling
# -------------------------------------------------------
class TimeSeriesScaler:
    def __init__(self):
        self.scalers = {}
    def fit_transform(self,X):
        N,L,F = X.shape
        Xs=np.zeros_like(X)
        for f in range(F):
            sc=MinMaxScaler((-1,1))
            flat=X[:,:,f].reshape(-1,1)
            sc.fit(flat)
            Xs[:,:,f]=sc.transform(flat).reshape(N,L)
            self.scalers[f]=sc
        return Xs
    def transform(self,X):
        N,L,F=X.shape
        Xs=np.zeros_like(X)
        for f in range(F):
            sc=self.scalers[f]
            Xs[:,:,f]=sc.transform(X[:,:,f].reshape(-1,1)).reshape(N,L)
        return Xs
    def transform_targets(self,y):
        sc=self.scalers[0]
        return sc.transform(y.reshape(-1,1)).reshape(len(y),-1)
    def inverse_transform_target(self,y_s):
        sc=self.scalers[0]
        return sc.inverse_transform(y_s.reshape(-1,1)).reshape(len(y_s),-1)

class SeqDataset(Dataset):
    def __init__(self,X,Y):
        self.X=X.astype('float32')
        self.Y=Y.astype('float32')
    def __len__(self):return len(self.X)
    def __getitem__(self,i):return self.X[i],self.Y[i]

# -------------------------------------------------------
# LSTM + Attention
# -------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self,input_dim,hid):
        super().__init__()
        self.lstm=nn.LSTM(input_dim,hid,batch_first=True)
    def forward(self,src):
        out,(h,c)=self.lstm(src)
        return out,h[-1],c[-1]

class Bahdanau(nn.Module):
    def __init__(self,enc_hid,dec_hid):
        super().__init__()
        self.W=nn.Linear(enc_hid+dec_hid,dec_hid)
        self.v=nn.Linear(dec_hid,1,bias=False)
    def forward(self,h_t,enc_out):
        batch,src_len,_=enc_out.shape
        h_rep=h_t.unsqueeze(1).repeat(1,src_len,1)
        energy=torch.tanh(self.W(torch.cat([h_rep,enc_out],dim=2)))
        att=self.v(energy).squeeze(2)
        return F.softmax(att,dim=1)

class Decoder(nn.Module):
    def __init__(self,enc_hid,dec_hid):
        super().__init__()
        self.att=Bahdanau(enc_hid,dec_hid)
        self.lcell=nn.LSTMCell(enc_hid+1,dec_hid)
        self.fc=nn.Linear(dec_hid,1)
    def forward(self,prev_y,h,c,enc_out):
        att=self.att(h,enc_out)
        ctx=torch.bmm(att.unsqueeze(1),enc_out).squeeze(1)
        inp=torch.cat([prev_y,ctx],dim=1)
        h,c=self.lcell(inp,(h,c))
        out=self.fc(h)
        return out,h,c,att

class Seq2Seq(nn.Module):
    def __init__(self,input_dim,enc_hid,dec_hid,device):
        super().__init__()
        self.enc=Encoder(input_dim,enc_hid)
        self.dec=Decoder(enc_hid,dec_hid)
        self.device=device
    def forward(self,x,horizon,tf_ratio,targets=None):
        enc_out,h,c=self.enc(x)
        batch=x.size(0)
        prev_y=torch.zeros(batch,1,device=self.device)
        outs=[];atts=[]
        for t in range(horizon):
            out,h,c,att=self.dec(prev_y,h,c,enc_out)
            outs.append(out)
            atts.append(att.unsqueeze(1))
            if targets is not None and random.random()<tf_ratio:
                prev_y=targets[:,t].unsqueeze(1)
            else:
                prev_y=out.detach()
        return torch.cat(outs,1).squeeze(-1), torch.cat(atts,1)

# -------------------------------------------------------
# FIXED Baseline (No errors)
# -------------------------------------------------------
def baseline_sarima_like(X_train,Y_train,X_test,horizon):
    train_series = X_train[:, -1, 0]
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model=SARIMAX(train_series,order=(1,1,1),seasonal_order=(1,1,1,12),
                      enforce_stationarity=False,enforce_invertibility=False)
        res=model.fit(disp=False)
        fc=res.forecast(horizon)
        return np.tile(fc.reshape(1,-1),(len(X_test),1))
    except Exception as e:
        print("SARIMA failed → using naive baseline.",e)
        last=train_series[-1]
        return np.tile(last,(len(X_test),horizon))

# -------------------------------------------------------
# Metrics
# -------------------------------------------------------
def rmse(a,b):return np.sqrt(mean_squared_error(a.reshape(-1),b.reshape(-1)))
def mape(a,b):
    denom=np.maximum(np.abs(a),1e-6)
    return np.mean(np.abs((a-b)/denom))*100

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device detected:",DEVICE)

    os.makedirs("models",exist_ok=True)
    os.makedirs("outputs",exist_ok=True)

    SEQ=60;H=12;ENC=64;DEC=64;BATCH=64;EPOCHS=20;LR=1e-3

    # LOAD DATA
    X_train,Y_train,X_val,Y_val,X_test,Y_test = get_splits(SEQ,H)
    print("Data shapes:",X_train.shape,X_val.shape,X_test.shape)

    # SCALE
    sc=TimeSeriesScaler()
    X_train_s=sc.fit_transform(X_train)
    X_val_s=sc.transform(X_val)
    X_test_s=sc.transform(X_test)
    Y_train_s=sc.transform_targets(Y_train)
    Y_val_s=sc.transform_targets(Y_val)
    Y_test_s=sc.transform_targets(Y_test)

    train_dl=DataLoader(SeqDataset(X_train_s,Y_train_s),batch_size=BATCH,shuffle=True)
    val_dl=DataLoader(SeqDataset(X_val_s,Y_val_s),batch_size=BATCH)
    test_dl=DataLoader(SeqDataset(X_test_s,Y_test_s),batch_size=BATCH)

    # MODEL
    model=Seq2Seq(3,ENC,DEC,DEVICE).to(DEVICE)
    opt=optim.Adam(model.parameters(),lr=LR)
    lossf=nn.MSELoss()

    best=float("inf")

    # TRAIN
    for ep in range(1,EPOCHS+1):
        model.train()
        total=0
        for xb,yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out,_=model(xb,H,0.5,yb)
            loss=lossf(out,yb)
            loss.backward()
            opt.step()
            total+=loss.item()*xb.size(0)
        tr_loss=total/len(X_train_s)

        # VAL
        model.eval()
        v=0
        with torch.no_grad():
            for xb,yb in val_dl:
                xb,yb=xb.to(DEVICE),yb.to(DEVICE)
                out,_=model(xb,H,0.0)
                v+=lossf(out,yb).item()*xb.size(0)
        v_loss=v/len(X_val_s)

        print(f"Epoch {ep}/{EPOCHS} Train={tr_loss:.6f} Val={v_loss:.6f}")
        if v_loss<best:
            best=v_loss
            torch.save(model.state_dict(),"models/best_seq2seq.pt")
            print("Saved best model.")

    # LOAD BEST
    model.load_state_dict(torch.load("models/best_seq2seq.pt",map_location=DEVICE))
    model.eval()

    # TEST PRED
    preds_s=[];atts=[]
    with torch.no_grad():
        for xb,yb in test_dl:
            xb=xb.to(DEVICE)
            out,att=model(xb,H,0.0)
            preds_s.append(out.cpu().numpy());atts.append(att.cpu().numpy())
    preds_s=np.vstack(preds_s)
    atts=np.vstack(atts)

    preds=sc.inverse_transform_target(preds_s)
    true=sc.inverse_transform_target(Y_test_s)

    # BASELINE
    base=baseline_sarima_like(X_train,Y_train,X_test,H)

    # METRICS
    rm=rmse(true,preds)
    mp=mape(true,preds)
    br=rmse(true,base)
    bm=mape(true,base)

    print("\n=== RESULTS ===")
    print("LSTM-Attn RMSE:",rm)
    print("LSTM-Attn MAPE:",mp)
    print("Baseline RMSE:",br)
    print("Baseline MAPE:",bm)

    # SAVE SUMMARY
    with open("outputs/summary.txt","w") as f:
        f.write(f"LSTM RMSE: {rm}\n")
        f.write(f"LSTM MAPE: {mp}\n")
        f.write(f"Baseline RMSE: {br}\n")
        f.write(f"Baseline MAPE: {bm}\n")

    # PLOTS
    plt.figure(figsize=(8,4))
    plt.plot(true[0],label="True")
    plt.plot(preds[0],label="Pred")
    plt.legend()
    plt.savefig("outputs/pred_sample.png")
    plt.close()

    plt.figure(figsize=(10,4))
    plt.imshow(atts[0],aspect='auto')
    plt.colorbar()
    plt.savefig("outputs/attention_sample.png")
    plt.close()

    # REPORT
    with open("report.txt","w") as r:
        r.write("Advanced Time Series Forecasting with LSTM + Bahdanau Attention\n")
        r.write("Author: Gorden Malcom\n\n")
        r.write(f"LSTM RMSE: {rm}\n")
        r.write(f"LSTM MAPE: {mp}\n")
        r.write(f"Baseline RMSE: {br}\n")
        r.write(f"Baseline MAPE: {bm}\n")
        r.write("Plots saved in outputs/.\n")

    print("\nAll done! File: report.txt is ready to submit.")

if __name__ == "__main__":
    main()
