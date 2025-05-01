import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

# Assumes fetch_data.py lives in src/data_ingestion
from spotify_client import SpotifyClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ▶ %(message)s"
)
logger = logging.getLogger(__name__)


def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory %s", path)


def save_json(data: Any, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved JSON to %s", out_path)


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and save Spotify user data via the API"
    )
    parser.add_argument(
        "--type",
        choices=[
            "top_tracks", "top_artists", "recently_played", "audio_features",
            "saved_tracks", "user_profile", "user_playlists", "playlist_tracks"
        ],
        required=True
    )
    parser.add_argument("--time_range",
                        choices=["short_term", "medium_term", "long_term"],
                        default="medium_term")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--track_ids", nargs="+",
                        help="Space-separated list of track IDs for audio_features")
    parser.add_argument("--playlist_id",
                        help="Playlist ID when fetching playlist_tracks")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / "data" / "raw" / "spotify_api"
    ensure_dir(out_dir)

    client = SpotifyClient()
    now = timestamp_str()

    if args.type in ("top_tracks", "top_artists"):
        kind = "tracks" if args.type == "top_tracks" else "artists"
        data = client.get_user_top_items(
            item_type=kind,
            time_range=args.time_range,
            limit=args.limit,
            offset=args.offset
        )
        fname = f"{args.type}_{args.time_range}_{args.limit}_{args.offset}_{now}.json"

    elif args.type == "recently_played":
        data = client.get_user_recently_played(limit=args.limit)
        fname = f"recently_played_{args.limit}_{now}.json"

    elif args.type == "saved_tracks":
        data = client.get_user_saved_tracks(
            limit=args.limit, offset=args.offset
        )
        fname = f"saved_tracks_{args.limit}_{args.offset}_{now}.json"

    elif args.type == "user_profile":
        data = client.get_user_profile()
        fname = f"user_profile_{now}.json"

    elif args.type == "user_playlists":
        data = client.get_user_playlists(
            limit=args.limit, offset=args.offset
        )
        fname = f"user_playlists_{args.limit}_{args.offset}_{now}.json"

    elif args.type == "playlist_tracks":
        if not args.playlist_id:
            parser.error("--playlist_id is required for playlist_tracks")
        data = client.get_playlist_tracks(
            playlist_id=args.playlist_id,
            limit=args.limit, offset=args.offset
        )
        fname = f"playlist_{args.playlist_id}_{args.limit}_{args.offset}_{now}.json"

    elif args.type == "audio_features":
        if not args.track_ids:
            parser.error("--track_ids is required for audio_features")
        data = client.get_audio_features(track_ids=args.track_ids)
        fname = f"audio_features_{len(args.track_ids)}_{now}.json"

    else:
        logger.error("Unsupported type: %s", args.type)
        return

    out_path = out_dir / fname
    save_json(data, out_path)


if __name__ == "__main__":
    main()
