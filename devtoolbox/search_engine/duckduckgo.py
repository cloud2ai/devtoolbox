import logging

from retry import retry

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException, TimeoutException

# Default number of search results
MAX_RESULTS = 5

# Duckduckgo API Rate Limit: 20 requests/second
# Retry delay in seconds
RETRY_DELAY_SECONDS = 5


class DuckDuckGoImageSearch(object):
    """
    Search engine for images

    Uses DuckDuckGo search engine API to search for images, supporting keyword
    search, region settings, safe search level, and image size filtering.
    """

    def __init__(
        self,
        keywords,
        region="us-en",
        safesearch="moderate",
        size="Large",
        max_results=MAX_RESULTS
    ):
        """
        Initialize the search engine

        Args:
            keywords (str): Search keywords
            region (str): Search region, default is "us-en"
            safesearch (str): Safe search level, options: "off", "moderate",
                "strict"
            size (str): Image size, default is "Large"
            max_results (int): Maximum number of results to return, default is
                MAX_RESULTS
        """
        self.keywords = keywords
        self.region = region
        self.safesearch = safesearch
        self.size = size
        self.max_results = max_results

    @retry((RatelimitException, TimeoutException),
           tries=3, delay=RETRY_DELAY_SECONDS)
    def search_image_urls(self):
        """
        Search for images and return a list of image URLs

        Uses DuckDuckGo API to search for images, automatically handling rate
        limit and timeout exceptions with up to 3 retries, each separated by
        RETRY_DELAY_SECONDS.

        Returns:
            list: List of image URLs
        """
        logging.info(f"Search ({self.max_results}) images "
                     f"with keywords [{self.keywords}]")
        results = DDGS().images(
            self.keywords,
            region=self.region,
            safesearch=self.safesearch,
            size=self.size,
            max_results=self.max_results
        )
        logging.debug(f"Search images: {results}")
        image_urls = [result["image"] for result in results]
        return image_urls
