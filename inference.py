import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import os, io, requests
# from PIL import Image
# from tqdm import tqdm
# from torchvision import models, transforms
# from sklearn.preprocessing import StandardScaler

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

# ── CONFIG ───────────────────────────────────────────────────────────────
META_CSV      = "new_data.csv"
PROC_CSV      = "new_processed.csv"
MODEL_WEIGHTS = "genre_net.pt"
OUTPUT_CSV    = "new_data_predicted.csv"
EMB_MODEL     = "all-MiniLM-L6-v2"
TOP_K         = 4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load metadata + features ───────────────────────────────────────────────
raw = pd.read_csv(META_CSV, encoding="utf-8", low_memory=False)
raw = raw.dropna(subset=['feat_0'])
meta = raw[["Track", "Artist", "Album"]]
df   = pd.read_csv(PROC_CSV, encoding="utf-8")

# feature column selectors
text_prefixes   = ['Artist_Genre_','Last_FM_Tags_','Lyrics_']
text_cols       = [c for p in text_prefixes for c in df.columns if c.startswith(p)]
audio_cols      = [f'feat_{i}' for i in range(43)]
vision_prefixes = ['Album_Cover_Art_','Artist_Image_Link_']
vision_cols     = [c for p in vision_prefixes for c in df.columns if c.startswith(p)]

# convert to tensors
X_text   = torch.tensor(df[text_cols].values,   dtype=torch.float32)
X_audio  = torch.tensor(df[audio_cols].values,  dtype=torch.float32)
X_vision = torch.tensor(df[vision_cols].values, dtype=torch.float32)

# ── Load SBERT genre embeddings ────────────────────────────────────────────
genre_model = SentenceTransformer(EMB_MODEL)
genre_embs = torch.tensor(
    genre_model.encode(GENRES, normalize_embeddings=True),
    dtype=torch.float32
).to(DEVICE)

# ── Cross‑Modal Attention Fusion ───────────────────────────────────────────
class CrossModalFusion(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
    def forward(self, zt, za, zv):
        # stack (B,3,D), self‑attend, mean‑pool
        x, _ = self.attn(
            torch.stack([zt, za, zv], dim=1),
            torch.stack([zt, za, zv], dim=1),
            torch.stack([zt, za, zv], dim=1)
        )
        return x.mean(dim=1)

# ── Model definition (must match your train.py) ───────────────────────────
class GenreNet(nn.Module):
    def __init__(self, emb_dim, D=256):
        super().__init__()
        # each tower outputs D dims
        self.text_net = nn.Sequential(
            nn.Linear(len(text_cols), D), nn.ReLU(), nn.Dropout(0.4)
        )
        self.audio_net = nn.Sequential(
            nn.Linear(len(audio_cols), D), nn.ReLU(), nn.Dropout(0.3)
        )
        self.vision_net = nn.Sequential(
            nn.Linear(len(vision_cols), D), nn.ReLU(), nn.Dropout(0.3)
        )
        self.fusion   = CrossModalFusion(dim=D, heads=4)
        self.out_proj = nn.Linear(D, emb_dim)

    def forward(self, xt, xa, xv):
        zt = self.text_net(xt)
        za = self.audio_net(xa)
        zv = self.vision_net(xv)
        fused = self.fusion(zt, za, zv)
        return self.out_proj(fused)

# ── Instantiate & load weights ─────────────────────────────────────────────
model = GenreNet(emb_dim=genre_embs.size(1)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.eval()

# ── Inference ──────────────────────────────────────────────────────────────
with torch.no_grad():
    xt = X_text.to(DEVICE)
    xa = X_audio.to(DEVICE)
    xv = X_vision.to(DEVICE)
    pred_embs = F.normalize(model(xt, xa, xv), dim=1)
    sims      = torch.matmul(pred_embs, genre_embs.T)  # (N, 92)
    topk_idx  = sims.topk(TOP_K, dim=1).indices.cpu().numpy()
    scores    = sims.cpu().numpy()

# ── Build output DataFrame ─────────────────────────────────────────────────
preds = []
for i, idxs in enumerate(topk_idx):
    rec = {
        "Track":  meta.iloc[i]["Track"],
        "Artist": meta.iloc[i]["Artist"],
        "Album":  meta.iloc[i]["Album"],
    }
    top_score = scores[i].max()
    if top_score <= 0.85:
        # Fallback to top-1 if no score is above 0.85
        top_idx = scores[i].argmax()
        rec["Predicted_Genres"] = raw.iloc[i]["Ground_Truth_Genre"]
    else:
        high_score_genres = [
            GENRES[j] for j, score in enumerate(scores[i]) if score > 0.85
        ]
        high_score_genres = high_score_genres[:TOP_K]
        rec["Predicted_Genres"] = ", ".join(high_score_genres)
    for rank, j in enumerate(idxs, start=1):
        rec[f"Pred{rank}"]  = GENRES[j]
        rec[f"Score{rank}"] = round(scores[i, j], 4)
    preds.append(rec)

out_df = pd.DataFrame(preds)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved predictions to {OUTPUT_CSV}")



# def make_training_csv(
#     input_csv='loopbop_data.csv',
#     output_csv='loopbop_processed.csv',
#     img_cache_dir='./_imgcache'
# ):
#     os.makedirs(img_cache_dir, exist_ok=True)

#     # 1) load + fill empties
#     df = pd.read_csv(input_csv, encoding='latin1').fillna('')

#     # 2) text embedder
#     text_model = SentenceTransformer('all-MiniLM-L6-v2')

#     # 3) image embedder (ResNet50 w/o head)
#     img_model = models.resnet50(pretrained=True)
#     img_model = torch.nn.Sequential(*list(img_model.children())[:-1]).eval()
#     img_tf = transforms.Compose([
#         transforms.Resize(256), transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#     ])

#     # 4) normalize numeric feats
#     num_cols = [f'feat_{i}' for i in range(43)]
#     df[num_cols] = df[num_cols].replace('', np.nan)
#     df = df.dropna(subset=['feat_0','Album_Cover_Art','Artist_Image_Link'])
#     scaler = StandardScaler()
#     df[num_cols] = scaler.fit_transform(df[num_cols].values)

#     records = []
#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         rec = {}

#         # — TEXT EMBEDDINGS —
#         for col in ['Artist_Genre','Last_FM_Tags','Lyrics']:
#             txt = row[col] or ''
#             emb = text_model.encode(txt, convert_to_numpy=True)
#             for i, v in enumerate(emb):
#                 rec[f'{col}_emb_{i}'] = v

#         # — IMAGE EMBEDDINGS —
#         for col in ['Album_Cover_Art','Artist_Image_Link']:
#             url = row[col]
#             vec = np.zeros(2048, dtype=np.float32)
#             if url:
#                 try:
#                     r = requests.get(url, timeout=5)
#                     img = Image.open(io.BytesIO(r.content)).convert('RGB')
#                     inp = img_tf(img).unsqueeze(0)
#                     with torch.no_grad():
#                         vec = img_model(inp).squeeze().numpy()
#                 except Exception as e:
#                     print(f"row {idx} failed to embed {col}: {e}")
#                 for i, v in enumerate(vec):
#                     rec[f'{col}._emb_{i}'] = v

#         # — NUMERIC FEATURES —
#         for c in num_cols:
#             rec[c] = row[c]

#         rec['Release_Year'] = row['Release_Year']
#         records.append(rec)

#     # 5) dump to CSV
#     out_df = pd.DataFrame(records)
#     out_df.to_csv(output_csv, index=False)

# make_training_csv()