import os
import requests
import pandas as pd
import numpy as np
import librosa
from youtube_search import YoutubeSearch
import yt_dlp
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from concurrent.futures import ThreadPoolExecutor
from yt_dlp.utils import DownloadError
from urllib.parse import quote_plus
import threading, time
import backoff
import soundfile as sf
import gc

from dotenv import load_dotenv
import os
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

def process_playlist(playlist_id, genre_name, download_dir='./audio', output_csv='data.csv'):
    #––– Setup Spotify client and fetch playlist items –––
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )
    )
    items = []
    results = sp.playlist_tracks(playlist_id)
    items.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        items.extend(results['items'])
    os.makedirs(download_dir, exist_ok=True)

    #––– Helper: Last.fm tags –––
    # global token bucket: 5 queries per second
    _lastfm_lock     = threading.Semaphore(5)
    _lastfm_interval = 1.0          # seconds
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def get_lastfm_tags(artist: str, track: str):
        """Thread‑safe, rate‑limited call to Last.fm."""
        with _lastfm_lock:
            try:
                params = {
                    'method': 'track.gettoptags',
                    'artist': artist,
                    'track':  track,
                    'api_key': LASTFM_API_KEY,
                    'format': 'json',
                    'autocorrect': 1,          # fixes slight misspellings
                }
                resp = requests.get('http://ws.audioscrobbler.com/2.0/', params=params, timeout=8)
                data = resp.json().get('toptags', {}).get('tag', [])
                return [t['name'] for t in data][:10]      # keep it compact
            except Exception as e:
                print(f"⚠  Last.fm error for {artist}-{track}: {e}")
                return []
            finally:
                # enforce <=5 req/sec no matter how many threads you spawn
                time.sleep(_lastfm_interval / 5.1)

    def get_lyrics(artist: str, track: str) -> str:
        """
        Try Lyrics.ovh, then LyricsFreak (scraped), else return ''.
        """
        try:
            # (a) lyrics.ovh – cheap JSON
            resp = requests.get(
                f"https://api.lyrics.ovh/v1/{quote_plus(artist)}/{quote_plus(track)}",
                timeout=8
            )
            txt = resp.json().get('lyrics', '')
            if txt.strip(): 
                return txt.strip()
        except Exception:
            pass

        try:
            # (b) quick scrape from lyricsfreak.com  (HTML)
            url = f"https://www.lyricsfreak.com/{quote_plus(artist.lower())}/{quote_plus(track.lower())}.html"
            html = requests.get(url, timeout=8).text
            if "lyrics" in html.lower():
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                verse = soup.select_one(".lyrictxt.js-lyrics.tospace")   # LF’s main class
                if verse:
                    return verse.get_text("\n").strip()
        except Exception:
            pass

        return ""


    #––– Helper: Librosa feature extraction –––
    def extract_features(path):
        """
        Load an audio file (ffmpeg must be installed to handle .webm/.mp3)
        and return a 43‑dimensional feature vector:
        - 13 MFCCs
        - 12 Chroma
        - 7 Spectral Contrast
        - 5 Scalars: spectral centroid, bandwidth, rolloff, tempo, zero crossing rate
        - 6 Tonnetz
        """

        # 1) Load (mono)
        y, sr = librosa.load(path, sr=22050, mono=True, duration=60.0)

        # 2) Multi‑dim features
        mfccs    = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        chroma   = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

        # 3) Scalar summaries (cast to float)
        centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
        tempo     = float(tempo)
        zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

        scalar_feats = np.array([centroid, bandwidth, rolloff, tempo, zcr], dtype=np.float32)

        # 4) Tonnetz
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)

        del y
        gc.collect()  # free memory

        # 5) Concatenate
        return np.concatenate([mfccs, chroma, contrast, scalar_feats, tonnetz])

    def process_track(item, sp, genre_name, download_dir):
        try:
            t = item.get('track') or {}
            name    = t.get('name') or '<unknown>'
            artists = [a['name'] for a in t.get('artists', [])] or ['<unknown>']
            artist  = artists[0]
            album   = t.get('album', {}).get('name', '')
            album_images = t.get('album', {}).get('images', [])
            album_cover = album_images[0]['url'] if album_images else ''
            spotify_url = t.get('external_urls', {}).get('spotify', '')
            release_date = t.get('album', {}).get('release_date', '')
            release_year = release_date.split('-')[0] if release_date else ''

            # –– Spotify artist details ––
            try:
                artist_info = sp.artist(t['artists'][0]['uri'])
                artist_genres = ";".join(artist_info.get('genres', []))
                artist_image  = (artist_info.get('images') or [{}])[0].get('url')
            except Exception as e:
                print(f"⚠  Spotify artist lookup failed for {artist}: {e}")
                artist_genres = artist_image = ''

            # –– Last.fm & lyrics ––
            try:
                tags = get_lastfm_tags(artist, name)
            except Exception as e:
                print(f"⚠  Last.fm failed for {name}: {e}"); tags = []
            lyrics = get_lyrics(artist, name)  # already wrapped

            # –– YouTube download ––
            audio_path = None
            try:
                query   = f"{artist} - {name}"
                y_hit   = YoutubeSearch(query, 1).to_dict()
                if y_hit:
                    vid_id = y_hit[0]['id']
                    yt_url = f"https://www.youtube.com/watch?v={vid_id}"
                    safe   = "".join(c if c.isalnum() else "_" for c in query)[:80]
                    y_opts = {
                        'format':'bestaudio/best',
                        'outtmpl':f"{download_dir}/{safe}.%(ext)s",
                        'retries':3,'fragment_retries':3,'timeout':15,
                        'postprocessors':[{'key':'FFmpegExtractAudio',
                                        'preferredcodec':'mp3',
                                        'preferredquality':'192'}],
                        "cookies_from_browser": ("chrome",),
                    }
                    with yt_dlp.YoutubeDL(y_opts) as ydl:
                        ydl.download([yt_url])
                    audio_path = f"{download_dir}/{safe}.mp3"
            except DownloadError as e:
                print(f"⚠  yt‑dlp download failed for {query}: {e}")
            except Exception as e:
                print(f"⚠  YouTube search failed for {name}: {e}")

            # –– Features ––
            def is_valid_audio(path):
                try:
                    with sf.SoundFile(path) as f:
                        return True
                except RuntimeError:
                    return False
            try:
                if audio_path and os.path.exists(audio_path) and is_valid_audio(audio_path):
                    feats = extract_features(audio_path)
                else:
                    print(f"⚠ Invalid or missing audio: {audio_path}")
                    feats = None
            except Exception as e:
                print(f"⚠  Librosa failed for {audio_path}: {e}")
                feats = None

            # –– Build row ––
            rec = dict(
                Track=name, Artist=";".join(artists), Album=album, Release_Year=release_year,
                Spotify_URL=spotify_url, Youtube_URL=yt_url ,Artist_Genre=artist_genres, 
                Last_FM_Tags=";".join(tags), Album_Cover_Art=album_cover,
                Artist_Image_Link=artist_image, Lyrics=lyrics, 
                Ground_Truth_Genre=genre_name,  # TODO: use genre_name
            )
            if feats is not None:
                rec.update({f'feat_{i}':v for i,v in enumerate(feats)})

            # Immediately clean up audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"⚠ Failed to delete {audio_path}: {e}")
            
            print(f"✔  {genre_name:<15} → '{name}' by {artist} ")
            return rec
        except Exception as e:
            print(f"⚠  process_track total failure for item: {e}")
            return None
        
    records = []
    with ThreadPoolExecutor(max_workers=16) as exe:
        futures = [exe.submit(process_track, it, sp, genre_name, download_dir)
                for it in items]
        for f in futures:
            try:
                rec = f.result()
                if rec: records.append(rec)
            except Exception as e:
                print(f"⚠  worker threw: {e}")

    try:
        df = pd.DataFrame(records)
        if os.path.isfile(output_csv):
            try:
                df_existing = pd.read_csv(output_csv)
                df = pd.concat([df_existing, df], ignore_index=True)
            except Exception as e:
                print(f"⚠  Could not read/concat {output_csv}: {e}")
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} tracks to {output_csv}")

    except Exception as e:
        print(f"⚠  FINAL write failed: {e}")

def first_public_playlist(
    genre: str,
    sp: spotipy.Spotify,
    market: str = "US",
    min_tracks: int = 20,
    min_follows: int = 5000,
    max_tracks: int = 50,
    page_size: int = 50,
    max_pages: int = 20,
):
    needle = genre.lower().replace(" ", "")
    query  = f'playlist:"{genre}"'

    for page in range(max_pages):
        offset = page * page_size
        hits = sp.search(q=query, type="playlist", limit=page_size,
                         offset=offset, market=market)["playlists"]["items"]

        if not hits:                       # no more results – stop early
            break
        for pl in hits:
            if pl is None or pl.get("public") is not True:
                continue
            if needle not in pl.get("name", "").lower().replace(" ", ""):
                continue
            if pl["tracks"]["total"] < min_tracks:
                continue
            if pl["tracks"]["total"] > max_tracks:
                continue
            meta = sp.playlist(pl["id"], market=market)
            if meta["followers"]["total"] < min_follows:
                continue
            print(f"✔  {genre:<15} → '{meta['name']}' "
                  f"({meta['tracks']['total']} tracks, "
                  f"{meta['followers']['total']} saves)\n   {meta['external_urls']['spotify']}")
            return pl["id"]

    raise RuntimeError(f"No suitable playlist found for “{genre}”.")

def process_genre(genre_name: str):
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
    )
    pid = first_public_playlist(genre_name, sp)
    process_playlist(pid, genre_name)

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

if __name__ == "__main__":
    for g in GENRES:          
        try:
            process_genre(g)
        except RuntimeError as e:
            print(f"⚠  {e}")