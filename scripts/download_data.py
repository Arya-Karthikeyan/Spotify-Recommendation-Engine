# downloads the tidytuesday spotify dataset into data/
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
out = Path(__file__).parent.parent / "data" / "spotify_songs.csv"
out.parent.mkdir(exist_ok=True)
urllib.request.urlretrieve(URL, out)
print("saved to", out)
