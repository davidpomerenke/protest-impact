from protest_impact.data.news.sources.google import search as google_search
from protest_impact.data.news.sources.mediacloud import search as mediacloud_search

engines = {
    "mediacloud": mediacloud_search,
    "google": google_search,
}
