import os
import time
import threading
import logging
from functools import lru_cache
import requests
import pandas as pd
import numpy as np
import librosa
from youtube_search import YoutubeSearch
import yt_dlp
from yt_dlp.utils import DownloadError
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus
import backoff
import soundfile as sf

# ── CONFIG ──
CLIENT_ID       = "6dad214ac6f249049d2ea16396e95533"
CLIENT_SECRET   = "2754466718b944c299de88ccad4ffb41"
LASTFM_API_KEY  = "9ba91375b3fa52ccffec116c0656f908"

# ── LOGGING ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── RATE LIMIT ──
_lastfm_lock     = threading.Semaphore(5)
_lastfm_interval = 1.0  # seconds

# ── YouTubeDL options (shared static part) ──
YDL_OPTS = {
    'format': 'bestaudio/best',
    'retries': 3,
    'fragment_retries': 3,
    'timeout': 15,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'cookies_from_browser': ('chrome',),
}

# ── HELPERS ──
def safe_remove(path, retries=3, backoff_s=0.1):
    for i in range(retries):
        try:
            os.remove(path)
            return
        except OSError as e:
            logging.warning(f"Remove failed ({e}), retry {i+1}/{retries}")
            time.sleep(backoff_s)
    logging.error(f"Could not delete {path} after {retries} tries")

@lru_cache(maxsize=512)
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def get_lastfm_tags(artist: str, track: str):
    with _lastfm_lock:
        params = {
            'method': 'track.gettoptags',
            'artist': artist,
            'track': track,
            'api_key': LASTFM_API_KEY,
            'format': 'json',
            'autocorrect': 1,
        }
        resp = requests.get('http://ws.audioscrobbler.com/2.0/', params=params, timeout=8)
        tags = resp.json().get('toptags', {}).get('tag', [])
        # enforce <=5 req/sec
        time.sleep(_lastfm_interval / 5.1)
        return [t['name'] for t in tags][:10]

@lru_cache(maxsize=512)
def get_lyrics(artist: str, track: str) -> str:
    # (a) Lyrics.ovh
    try:
        resp = requests.get(
            f"https://api.lyrics.ovh/v1/{quote_plus(artist)}/{quote_plus(track)}",
            timeout=8
        )
        txt = resp.json().get('lyrics', '').strip()
        if txt:
            return txt
    except Exception:
        pass

    # (b) LyricsFreak scrape
    try:
        url = f"https://www.lyricsfreak.com/{quote_plus(artist.lower())}/{quote_plus(track.lower())}.html"
        html = requests.get(url, timeout=8).text
        if "lyrics" in html.lower():
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            verse = soup.select_one(".lyrictxt.js-lyrics.tospace")
            if verse:
                return verse.get_text("\n").strip()
    except Exception:
        pass

    return ""

def extract_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=None, mono=True)
    mfccs    = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma   = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    scalar_feats = np.array([centroid, bandwidth, rolloff, tempo, zcr], dtype=np.float32)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)
    return np.concatenate([mfccs, chroma, contrast, scalar_feats, tonnetz])

# ── CORE ──
def process_track(item: dict, sp: spotipy.Spotify, download_dir: str) -> dict:
    try:
        t = item.get('track') or {}
        name    = t.get('name') or '<unknown>'
        artists = [a['name'] for a in t.get('artists', [])] or ['<unknown>']
        artist  = artists[0]

        # Spotify metadata
        try:
            artist_info = sp.artist(t['artists'][0]['uri'])
            artist_genres = ";".join(artist_info.get('genres', []))
            artist_image  = (artist_info.get('images') or [{}])[0].get('url', '')
        except Exception as e:
            logging.error(f"Spotify lookup failed for {artist}: {e}")
            artist_genres = artist_image = ''

        # Last.fm tags & lyrics
        try:
            tags = get_lastfm_tags(artist, name)
        except Exception as e:
            logging.error(f"Last.fm failed for {name}: {e}")
            tags = []
        lyrics = get_lyrics(artist, name)

        # YouTube download
        query = f"{artist} - {name}"
        y_hit = YoutubeSearch(query, 1).to_dict()
        audio_path = None
        if y_hit:
            vid_id = y_hit[0]['id']
            yt_url = f"https://www.youtube.com/watch?v={vid_id}"
            safe_name = "".join(c if c.isalnum() else "_" for c in query)[:80]
            opts = dict(YDL_OPTS, outtmpl=f"{download_dir}/{safe_name}.%(ext)s")
            try:
                with yt_dlp.YoutubeDL(opts) as ydl_inst:
                    ydl_inst.download([yt_url])
                audio_path = f"{download_dir}/{safe_name}.mp3"
            except DownloadError as e:
                logging.error(f"yt-dlp failed for {query}: {e}")

        # Feature extraction
        feats = None
        if audio_path and os.path.exists(audio_path):
            try:
                with sf.SoundFile(audio_path):
                    pass  # validate file
                feats = extract_features(audio_path)
            except Exception as e:
                logging.error(f"Feature extraction failed for {audio_path}: {e}")

        # Build record
        rec = {
            'Track': name,
            'Artist': ";".join(artists),
            'Album': t.get('album', {}).get('name', ''),
            'Release_Year': t.get('album', {}).get('release_date', '').split('-')[0] or '',
            'Spotify_URL': t.get('external_urls', {}).get('spotify', ''),
            'YouTube_URL': f"https://www.youtube.com/watch?v={vid_id}" if y_hit else '',
            'Artist_Genre': artist_genres,
            'Last_FM_Tags': ";".join(tags),
            'Album_Cover_Art': (t.get('album', {}).get('images') or [{}])[0].get('url', ''),
            'Artist_Image_Link': artist_image,
            'Lyrics': lyrics,
        }
        if feats is not None:
            rec.update({f'feat_{i}': float(v) for i, v in enumerate(feats)})

        # Cleanup
        if audio_path and os.path.exists(audio_path):
            safe_remove(audio_path)

        return rec

    except Exception as e:
        logging.error(f"process_track total failure: {e}")
        return None

def process_playlist(playlist_id: str, download_dir: str = './audio', output_csv: str = 'new_data.csv'):
    # Spotify setup
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )
    )
    items = []
    results = sp.playlist_tracks(playlist_id)
    items.extend(results['items'])
    while results.get('next'):
        results = sp.next(results)
        items.extend(results['items'])

    os.makedirs(download_dir, exist_ok=True)
    batch_size = 200

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(2600, len(items), batch_size):
            chunk = items[i:i+batch_size]
            futures = [executor.submit(process_track, it, sp, download_dir) for it in chunk]
            batch = [f.result() for f in futures if f.result()]

            if batch:
                df = pd.DataFrame(batch)
                write_header = not os.path.isfile(output_csv)
                df.to_csv(output_csv, mode='a', index=False, header=write_header)
                logging.info(f"Saved {len(batch)} tracks (batch {i//batch_size + 1})")

if __name__ == "__main__":
    process_playlist("4PUXZ8keC8oz1h634oAKb7")