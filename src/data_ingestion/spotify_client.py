import os
import time
import logging
from typing import List, Dict, Any, Optional

from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# Configure module‐level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ▶ %(message)s"
)
logger = logging.getLogger(__name__)


class SpotifyClient:
    """
    Wrapper around spotipy.Spotify for common user‐data ingestion tasks.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        scope: str = (
            "user-top-read user-read-recently-played "
            "user-library-read playlist-read-private"
        )
    ):
        self.client_id = client_id or os.getenv("SPOTIPY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("SPOTIPY_REDIRECT_URI")
        self.scope = scope

        if not all([self.client_id, self.client_secret, self.redirect_uri]):
            raise ValueError("Must set SPOTIPY_CLIENT_ID, CLIENT_SECRET, and REDIRECT_URI")

        auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=self.scope,
            cache_path=".spotify_token_cache"
        )

        self.sp = Spotify(auth_manager=auth_manager)

        cc_manager = SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        self.sp_client = Spotify(client_credentials_manager=cc_manager)

        logger.info("Authenticated to Spotify with scope=%s", self.scope)

    def get_user_profile(self) -> Dict[str, Any]:
        """Fetch current user’s profile (display name, country, followers, etc.)."""
        logger.info("Fetching user profile")
        return self.sp.current_user()

    def get_user_top_items(
        self,
        item_type: str = "tracks",
        time_range: str = "medium_term",
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch the user’s top tracks or artists.
        """
        assert item_type in ("tracks", "artists")
        logger.info(
            "Fetching top %s (range=%s, limit=%d, offset=%d)",
            item_type, time_range, limit, offset
        )
        try:
            method = getattr(self.sp, f"current_user_top_{item_type}")
            return method(time_range=time_range, limit=limit, offset=offset)
        except SpotifyException as e:
            logger.error("Error fetching top %s: %s", item_type, e, exc_info=True)
            raise

    def get_user_recently_played(
        self,
        limit: int = 50,
        after: Optional[int] = None,
        before: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fetch the user’s recently played tracks.
        """
        logger.info("Fetching recently played (limit=%d, after=%s, before=%s)",
                    limit, after, before)
        try:
            return self.sp.current_user_recently_played(
                limit=limit, after=after, before=before
            )
        except SpotifyException as e:
            logger.error("Error fetching recently played: %s", e, exc_info=True)
            raise

    def get_user_saved_tracks(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch the user’s “Liked Songs” library.
        """
        logger.info("Fetching saved tracks (limit=%d, offset=%d)", limit, offset)
        try:
            return self.sp.current_user_saved_tracks(limit=limit, offset=offset)
        except SpotifyException as e:
            logger.error("Error fetching saved tracks: %s", e, exc_info=True)
            raise

    def get_user_playlists(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch playlists owned/followed by the user.
        """
        logger.info("Fetching user playlists (limit=%d, offset=%d)", limit, offset)
        try:
            return self.sp.current_user_playlists(limit=limit, offset=offset)
        except SpotifyException as e:
            logger.error("Error fetching playlists: %s", e, exc_info=True)
            raise

    def get_playlist_tracks(
        self,
        playlist_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch items in a specific playlist.
        """
        logger.info(
            "Fetching tracks for playlist %s (limit=%d, offset=%d)",
            playlist_id, limit, offset
        )
        try:
            return self.sp.playlist_items(playlist_id, limit=limit, offset=offset)
        except SpotifyException as e:
            logger.error("Error fetching playlist %s: %s", playlist_id, e, exc_info=True)
            raise

    def get_audio_features(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        import requests

        try:
            tokinfo = self.sp_client._auth_manager.get_cached_token()
            access_token = tokinfo["access_token"]
        except Exception:
            # older versions support as_dict=False
            access_token = self.sp_client._auth_manager.get_access_token(as_dict=False)
        headers = {"Authorization": f"Bearer {access_token}"}

        features: List[Dict[str, Any]] = []
        for tid in track_ids:
            url = f"https://api.spotify.com/v1/audio-features/{tid}"
            try:
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                feat = resp.json()
                features.append(feat)
            except Exception as e:
                logger.warning("Failed to fetch audio_features for %s: %s", tid, e)
        return features