import os, io, requests, random, math
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True 
from torch.utils.data import WeightedRandomSampler

GENRES = [
    "Alternative Pop","City Pop","Dream Pop","Electropop","Indie Pop","Dance Pop","Hyperpop",
    "Sunshine Pop","Bubblegum Pop","J-Pop","K-Pop","C-Pop","Europop",
    "Bedroom Pop","Synth Pop","Latin Pop","Yacht Rock","Soft Rock",

    "Alternative Rock","Garage Rock","Indie Rock","Metal","New Wave","Post-Punk",
    "Progressive Rock","Psychedelic Rock","Punk Rock","Shoegaze","Pop Punk","Surf Rock",
    "Hard Rock","Rock 'n' Roll","Grunge","Glam Rock",

    "Boom Bap","Trap","Rage","Jazz Rap","Trap Soul","Pop Rap","Drill","Cloud Rap","G-Funk",
    "Contemporary R&B","Neo Soul","Soul","Psychedelic Soul","Slow Jams","Disco","New Jack Swing",

    "Electronica","Eurodance","Future Bass","House","Jersey Club","Nu Disco","Synthwave",
    "Techno","Trance","UK Garage","Drum and Bass","Dubstep","Hardstyle", "Lo-Fi","Industrial","Ambient",
    
    "Trip Hop","Country","Bluegrass","Folk","Cool Jazz","Bebop","Jazz Fusion",
    "Gospel","Blues","Bachata","Corridos tumbados","Bossa Nova",
    "Baile Funk","Reggae","Dancehall","Afrobeats","Amapiano",
    "Pop", "Hip Hop","Electronic","R&B","Jazz"
]

# CONFIG
SEED        = 66
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RAW_CSV     = 'data.csv'    
PROC_CSV    = 'processed.csv'        
IMG_CACHE   = './_imgcache'        
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 0.001
EMB_MODEL   = 'all-MiniLM-L6-v2'

# LOAD DATA
df = pd.read_csv(PROC_CSV, encoding='utf-8', low_memory=False)

text_prefixes   = ['Artist_Genre_','Last_FM_Tags_','Lyrics_']
text_cols       = [c for p in text_prefixes for c in df.columns if c.startswith(p)]
audio_cols      = [f'feat_{i}' for i in range(43)]
vision_prefixes = ['Album_Cover_Art_', 'Artist_Image_Link_']
vision_cols     = [c for p in vision_prefixes for c in df.columns if c.startswith(p)]
label_cols      = [c for c in df.columns if c.startswith('Ground_Truth_Genre_emb_')]

# coerce everything you expect to be numeric
for col in label_cols + text_cols + audio_cols + vision_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# now drop any rows where *any* of those went to NaN
df.dropna(subset=label_cols + text_cols + audio_cols + vision_cols, inplace=True)
df.reset_index(drop=True, inplace=True)

# finally you can safely extract numpy arrays
X_text   = df[text_cols].values.astype(np.float32)
X_audio  = df[audio_cols].values.astype(np.float32)
X_vision = df[vision_cols].values.astype(np.float32)
Y_emb    = df[label_cols].values.astype(np.float32)

genre_to_idx = {g:i for i,g in enumerate(GENRES)}
label_idx    = df['Ground_Truth_Genre'].map(genre_to_idx).values
vc = pd.Series(label_idx).value_counts()
problem_idxs = vc[vc < 2].index.tolist()

for i in problem_idxs:
    genre = df['Ground_Truth_Genre'][label_idx == i].iloc[0]
    print(f"'{genre}' (class {i}) has only {vc[i]} sample(s)")

valid_idx = np.isin(label_idx, vc[vc >= 2].index)
df = df[valid_idx].reset_index(drop=True)
label_idx = label_idx[valid_idx]
Y_emb = Y_emb[valid_idx]

train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.10, random_state=SEED, stratify=label_idx
)

class GenreDataset(Dataset):
    def __init__(self, xt, xa, xv, y):
        self.xt, self.xa, self.xv, self.y = xt, xa, xv, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (self.xt[idx], self.xa[idx], self.xv[idx], self.y[idx])

train_ds = GenreDataset(X_text[train_idx],  X_audio[train_idx],  X_vision[train_idx],  Y_emb[train_idx])
test_ds  = GenreDataset(X_text[test_idx],   X_audio[test_idx],   X_vision[test_idx],   Y_emb[test_idx])

weights = 1.0 / torch.bincount(torch.tensor(label_idx))
sampler = WeightedRandomSampler(weights[label_idx[train_idx]], len(train_idx), replacement=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# GENRE EMBEDDING MATRIX (90 × 384)
genre_model = SentenceTransformer(EMB_MODEL)
genre_embs  = torch.tensor(
    genre_model.encode(GENRES, normalize_embeddings=True),
    dtype=torch.float32, device=DEVICE
)

class CrossModalFusion(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, zt, za, zv):
        # zt, za, zv are all (B, D) → stack as (B, 3, D)
        x, _ = self.attn(torch.stack([zt, za, zv], dim=1),
                         torch.stack([zt, za, zv], dim=1),
                         torch.stack([zt, za, zv], dim=1))
        return x.mean(dim=1)  # (B, D)

# MODEL
class GenreNet(nn.Module):
    def __init__(self, emb_dim, D=256):
        super().__init__()
        # adjust all towers to output same D
        self.text_net = nn.Sequential(
            nn.Linear(len(text_cols), D), nn.ReLU(), nn.Dropout(0.4),
        )
        self.audio_net = nn.Sequential(
            nn.Linear(len(audio_cols), D), nn.ReLU(), nn.Dropout(0.3),
        )
        self.vision_net = nn.Sequential(
            nn.Linear(len(vision_cols), D), nn.ReLU(), nn.Dropout(0.3),
        )
        self.fusion      = CrossModalFusion(dim=D, heads=4)
        self.out_proj    = nn.Linear(D, emb_dim)

    def forward(self, xt, xa, xv):
        zt = self.text_net(xt)
        za = self.audio_net(xa)
        zv = self.vision_net(xv)
        fused = self.fusion(zt, za, zv)
        return self.out_proj(fused)

model = GenreNet(emb_dim=genre_embs.size(1)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# TRAINING
def cosine_loss(pred, target):
    pred   = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    return 1 - (pred * target).sum(1).mean()

for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for xt, xa, xv, y in train_loader:
        xt, xa, xv, y = xt.to(DEVICE), xa.to(DEVICE), xv.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xt, xa, xv)
        loss = cosine_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running += loss.item() * xt.size(0)
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch}/{EPOCHS} - loss {running/len(train_ds):.4f}')

# EVALUATION
model.eval()
correct_top1 = 0
correct_top3 = 0
correct_top5 = 0
tot = 0

with torch.no_grad():
    for xt, xa, xv, y_true in test_loader:
        xt, xa, xv = xt.to(DEVICE), xa.to(DEVICE), xv.to(DEVICE)
        song_emb = model(xt, xa, xv)
        song_emb = F.normalize(song_emb, dim=1)
        sims = torch.matmul(song_emb, genre_embs.T)
        top3 = sims.topk(3, dim=1).indices.cpu().numpy()
        top5 = sims.topk(5, dim=1).indices.cpu().numpy()
        true_idx = [genre_to_idx[g] for g in df['Ground_Truth_Genre'].iloc[test_idx[tot:tot+len(xt)]]]
        for j, t in enumerate(true_idx):
            if top3[j,0] == t:
                correct_top1 += 1
                correct_top3 += 1
                correct_top5 += 1
            elif t in top3[j]:
                correct_top3 += 1
                correct_top5 += 1
            elif t in top5[j]:
                correct_top5 += 1
        tot += len(xt)

top1_acc = correct_top1 / tot
top3_acc = correct_top3 / tot
top5_acc = correct_top5 / tot
print(f'\nTop‑1 accuracy  : {top1_acc:.3f}')
print(f'Top‑3 accuracy  : {top3_acc:.3f}')
print(f'Top‑5 accuracy  : {top5_acc:.3f}')

torch.save(model.state_dict(), 'genre_net.pt')
print('model saved to genre_net.pt')