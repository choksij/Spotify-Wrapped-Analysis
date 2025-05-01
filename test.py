from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials

# 1) Try with your user OAuth client:
sp_user = Spotify(
    auth_manager=SpotifyOAuth(
        client_id="082ef094e9c4458f85a8bf8773846e09",
        client_secret="bb9853a873bd4e7587e276444f17f1d5",
        redirect_uri="http://localhost:8888/callback",
        scope="user-read-private",  # no extra scopes needed for audio-features
        cache_path=".spotipyoauthcache"
    )
)
print("User-auth token:", sp_user.auth_manager.get_access_token(as_dict=False)[:10], "...")

# 2) Try with client-credentials flow:
cc = SpotifyClientCredentials(
    client_id="082ef094e9c4458f85a8bf8773846e09",
    client_secret="bb9853a873bd4e7587e276444f17f1d5"
)
sp_cc = Spotify(client_credentials_manager=cc)
token = cc.get_access_token()
print("CC token:", token[:10], "...")

# 3) Pick one of your track IDs and try both:
test_id = "1jKXjxMWlq4BhH6f9GtZbu"   # replace if you know another
print("user-auth call →", sp_user.audio_features([test_id]))
print("cc-flow call   →", sp_cc.audio_features([test_id]))
