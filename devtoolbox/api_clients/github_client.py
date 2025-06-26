"""
GitHub API Client Module

This module provides functionality to interact with GitHub's trending
repositories and repository data. It includes web scraping of GitHub's
trending page and GitHub API integration for detailed repository
information.

Classes:
    GithubHandler: Main handler for GitHub trending repositories
    Repo: Repository object wrapper with detailed information

Dependencies:
    - PyGithub: GitHub API client
    - requests: HTTP client for web scraping
    - BeautifulSoup: HTML parsing
"""

import datetime
import logging
import os

from bs4 import BeautifulSoup
from github import Auth, Github
import requests

# Get logger for this module
logger = logging.getLogger(__name__)

# Maximum number of trending repositories to fetch
MAX_TRENDINGS_NUM = 25

# GitHub base URLs and selectors for web scraping
BASE_URL = "https://github.com"
TRENDING_URL = "%s/trending" % BASE_URL
REPOS_XPATH = "article.Box-row"
REPO_XPATH = "h2.h3.lh-condensed > a"

# Query parameter constants for filtering
LANG_CODE_ANY = "any"
LANGUAGE_ANY = "any"
SINCE_TODAY = "today"

# README file names to search for
README_MDS = ["readme.md", "readme.rst"]

# Thresholds for filtering
MAIN_LANGUAGE_THRESHOLD = 0.25
CONTRIBUTION_THRESHOLD = 0.1

# Issue status constants
ISSUE_OPEN = "open"

# Environment variable name for GitHub authentication token
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"


class GithubHandler(object):
    """
    GitHub Trending Repositories Handler

    This class handles fetching and processing of GitHub's trending
    repositories. It combines web scraping of GitHub's trending page
    with GitHub API calls to provide comprehensive repository
    information.

    Attributes:
        token (str): GitHub authentication token
        _github (Github): PyGithub client instance
    """

    def __init__(self, token=None):
        """
        Initialize GitHub handler with authentication token

        Args:
            token (str, optional): GitHub personal access token. If not provided,
                                 will try to get from GITHUB_TOKEN environment variable.

        Raises:
            ValueError: If no token is provided or found in environment
        """
        logger.info("Initializing GithubHandler")

        # Get token from parameter or environment variable
        self.token = token or os.getenv(GITHUB_TOKEN_ENV, "")

        if not self.token:
            logger.error("GitHub token not provided")
            raise ValueError(
                "GitHub token is required. Please provide it as a parameter "
                "or set the GITHUB_TOKEN environment variable.")

        logger.info("GitHub token configured successfully")
        self._github = None

    @property
    def github(self):
        """
        Lazy-loaded GitHub API client

        Returns:
            Github: Authenticated PyGithub client instance
        """
        if not self._github:
            logger.debug("Creating GitHub client with authentication")
            auth = Auth.Token(self.token)
            self._github = Github(auth=auth)
            logger.debug("GitHub client created successfully")

        return self._github

    def get_trendings(self, lang_code=LANG_CODE_ANY, lang=LANGUAGE_ANY,
                      since=SINCE_TODAY, num=MAX_TRENDINGS_NUM):
        """
        Fetch trending repositories from GitHub

        This method scrapes GitHub's trending page and enriches the data
        with GitHub API calls to get detailed repository information.

        Args:
            lang_code (str): Filter by spoken language (e.g., 'en', 'zh')
            lang (str): Filter by programming language (e.g., 'python', 'javascript')
            since (str): Time range filter ('today', 'weekly', 'monthly')
            num (int): Maximum number of repositories to return

        Returns:
            list: List of Repo objects with detailed information

        Raises:
            requests.RequestException: If HTTP request fails
        """
        logger.info("Starting to fetch trending repositories with params: "
                   "lang_code=%s, lang=%s, since=%s, num=%s",
                   lang_code, lang, since, num)

        query_trending_url = self._build_query_url(
            lang_code, lang, since)

        logger.info("Query github trending url: %s" % query_trending_url)

        try:
            logger.debug("Making HTTP request to GitHub trending page")
            response = requests.get(query_trending_url)
            response.raise_for_status()
            logger.debug("HTTP request successful, status code: %s",
                        response.status_code)
        except requests.RequestException as e:
            logger.error("Failed to fetch trending page: %s", e)
            raise

        logger.debug("Parsing HTML content with BeautifulSoup")
        soup = BeautifulSoup(response.text, "html.parser")
        repo_items = soup.select(REPOS_XPATH)
        logger.info("Found %d repository items on trending page",
                   len(repo_items))

        repos = []
        processed_count = 0
        skipped_count = 0

        for idx, repo in enumerate(repo_items):
            if idx + 1 > num:
                logger.debug("Reached maximum number of repos (%d)", num)
                break

            try:
                repo_xpath = repo.select_one(REPO_XPATH)
                if not repo_xpath:
                    logger.warning("No repo link found for item %d", idx)
                    skipped_count += 1
                    continue

                repo_url = self._clean(repo_xpath["href"])
                logger.debug("Processing repo: %s", repo_url)

                repo = Repo(self.github, repo_url)
                if not repo.readme:
                    logger.warning("Skip project %s due to readme is not found",
                                  repo_url)
                    skipped_count += 1
                    continue

                repos.append(repo)
                processed_count += 1
                logger.debug("Successfully processed repo: %s", repo_url)

            except Exception as e:
                logger.error("Failed to process repo item %d: %s", idx, e)
                skipped_count += 1
                continue

        logger.info("Trending repos processing completed: %d processed, "
                   "%d skipped, %d total", processed_count, skipped_count,
                   len(repos))
        return repos

    def _build_query_url(self, lang_code, lang, since):
        """
        Build GitHub trending page URL with filters

        Args:
            lang_code (str): Spoken language filter
            lang (str): Programming language filter
            since (str): Time range filter

        Returns:
            str: Complete GitHub trending page URL with filters
        """
        logger.debug("Building query URL with params: lang_code=%s, "
                    "lang=%s, since=%s", lang_code, lang, since)

        query_trending_url = TRENDING_URL

        if not lang == LANGUAGE_ANY:
            query_trending_url = "%s/%s" % (query_trending_url, lang)

        query_trending_url = "%s?since=%s" % (query_trending_url, since)

        if not lang_code == LANG_CODE_ANY:
            query_trending_url = "%s&spoken_language_code=%s" % (
                query_trending_url, lang_code)

        logger.debug("Built query URL: %s", query_trending_url)
        return query_trending_url

    def _clean(self, text):
        """
        Clean repository URL from HTML href attribute

        Args:
            text (str): Raw href text from HTML

        Returns:
            str: Cleaned repository path
        """
        if not text:
            logger.warning("Received empty text in _clean method")
            return ""

        if text[0] == "/":
            cleaned = text[1:]
            logger.debug("Cleaned text: '%s' -> '%s'", text, cleaned)
            return cleaned
        return text


class Repo(object):
    """
    GitHub Repository Object Wrapper

    This class provides a wrapper around GitHub repository data, combining
    information from GitHub's trending page with detailed API data. It includes
    repository metadata, README content, language statistics, and contributor
    information.

    Attributes:
        github (Github): PyGithub client instance
        path (str): Repository path (owner/repo)
        _repo (Repository): Lazy-loaded PyGithub Repository object
        _readme (str): Cached README content
        _main_languages (list): Cached main programming languages
        _contributors (list): Cached repository contributors
        _main_contributors (list): Cached main contributors
        _images (list): Associated images for the repository
    """

    def __init__(self, github, path):
        """
        Initialize repository object with GitHub client and path

        Args:
            github (Github): PyGithub client instance
            path (str): Repository path in format 'owner/repo'
        """
        logger.debug("Initializing Repo object for path: %s", path)
        self.github = github
        self.path = path

        self._repo = None
        self._readme = None
        self._main_languages = []

        self._contributors = []
        self._main_contributors = []

        self._images = []

        self._copy_attrs()
        logger.debug("Repo object initialized successfully for: %s", path)

    @property
    def repo(self):
        """
        Lazy-loaded PyGithub Repository object

        This property ensures the repository data is only fetched once
        and cached for subsequent access.

        Returns:
            Repository: PyGithub Repository object

        Raises:
            Exception: If repository cannot be fetched from GitHub API
        """
        if not self._repo:
            logger.debug("Fetching repository data for: %s", self.path)
            try:
                self._repo = self.github.get_repo(self.path)
                logger.debug("Repository data fetched successfully for: %s",
                           self.path)
            except Exception as e:
                logger.error("Failed to fetch repository data for %s: %s",
                           self.path, e)
                raise

        return self._repo

    def _copy_attrs(self):
        """
        Copy attributes from GitHub Repository raw data

        This method copies all attributes from the GitHub API response
        to the Repo object instance for easy access.

        Raises:
            Exception: If repository data cannot be accessed
        """
        logger.debug("Copying attributes from GitHub repository: %s",
                    self.path)
        try:
            attrs = self.repo.raw_data
            logger.debug("Repository attributes: %s", attrs)
            for attr, value in attrs.items():
                setattr(self, attr, value)
            logger.debug("Attributes copied successfully for: %s", self.path)
        except Exception as e:
            logger.error("Failed to copy attributes for %s: %s",
                        self.path, e)
            raise

    @property
    def readme(self):
        """
        Repository README content

        This property fetches and caches the README content from the repository.
        It searches for common README file names and decodes the content.

        Returns:
            str: README content as UTF-8 string, empty string if not found
        """
        if not self._readme:
            logger.debug("Fetching README for repository: %s", self.path)
            readme_path = self._get_readme_path()
            if readme_path:
                try:
                    logger.debug("Found README at path: %s", readme_path)
                    content = self.repo.get_contents(readme_path)
                    self._readme = content.decoded_content.decode('utf-8')
                    logger.debug("README content loaded successfully, "
                               "length: %d characters", len(self._readme))
                except Exception as e:
                    logger.error("Failed to load README for %s: %s",
                               self.path, e)
                    self._readme = ""
            else:
                logger.warning("No README file found for repository: %s",
                             self.path)
                self._readme = ""

        return self._readme

    @property
    def main_languages(self):
        """
        Main programming languages used in the repository

        This property analyzes the repository's language statistics and
        returns languages that exceed the threshold percentage of total code.

        Returns:
            list: List of main programming language names
        """
        if not self._main_languages:
            logger.debug("Fetching languages for repository: %s", self.path)
            try:
                languages = self.repo.get_languages()
                total_lines = sum(languages.values())
                logger.debug("Repository %s has %d total lines of code",
                           self.path, total_lines)

                main_languages = []
                for language, lines in languages.items():
                    percentage = lines / total_lines
                    logger.debug("Language %s: %d lines (%.2f%%)",
                               language, lines, percentage * 100)
                    if percentage > MAIN_LANGUAGE_THRESHOLD:
                        main_languages.append(language)
                        logger.debug("Added %s as main language", language)

                self._main_languages = main_languages
                logger.info("Main languages for %s: %s", self.path,
                           main_languages)
            except Exception as e:
                logger.error("Failed to fetch languages for %s: %s",
                           self.path, e)
                self._main_languages = []

        return self._main_languages

    @property
    def contributors_count(self):
        """
        Total number of contributors to the repository

        Returns:
            int: Number of contributors, 0 if cannot be determined
        """
        try:
            count = self.repo.get_contributors().totalCount
            logger.debug("Contributors count for %s: %d", self.path, count)
            return count
        except Exception as e:
            logger.error("Failed to get contributors count for %s: %s",
                       self.path, e)
            return 0

    @property
    def contributors(self):
        """
        All contributors to the repository

        This property fetches and caches the list of all contributors
        with their contribution statistics.

        Returns:
            list: List of Contributor objects from PyGithub
        """
        if not self._contributors:
            logger.debug("Fetching contributors for repository: %s",
                        self.path)
            try:
                self._contributors = list(self.repo.get_contributors())
                logger.debug("Fetched %d contributors for %s",
                           len(self._contributors), self.path)
            except Exception as e:
                logger.error("Failed to fetch contributors for %s: %s",
                           self.path, e)
                self._contributors = []
        return self._contributors

    @property
    def main_contributors(self):
        """
        Main contributors based on contribution percentage

        This property analyzes contributor statistics and returns
        contributors who exceed the threshold percentage of total contributions.

        Returns:
            list: List of main Contributor objects
        """
        if not self._main_contributors:
            logger.debug("Calculating main contributors for: %s", self.path)
            contributors = []
            total_contributions = sum(c.contributions for c in
                                    self.contributors)
            logger.debug("%s total contributions: %s" % (
                self.path, total_contributions))

            for c in self.contributors:
                contributor_ratio = c.contributions / total_contributions
                logger.debug("%s contributions ratio: %s" % (
                    c.name, contributor_ratio))
                if contributor_ratio > CONTRIBUTION_THRESHOLD:
                    contributors.append(c)
                    logger.debug("Added %s as main contributor", c.name)

            self._main_contributors = contributors
            logger.info("Main contributors for %s: %s", self.path,
                       [c.name for c in contributors])

        return self._main_contributors

    def _get_readme_path(self):
        """
        Find README file path in repository root

        This method searches the repository root directory for common
        README file names and returns the first match found.

        Returns:
            str: README file path, None if not found
        """
        logger.debug("Searching for README file in repository: %s",
                    self.path)
        try:
            contents = self.repo.get_contents("")
            logger.debug("Repository files: %s", contents)

            readme_path = None
            for content in contents:
                if content.path.lower() in README_MDS:
                    readme_path = content.path
                    logger.debug("Found README file: %s", readme_path)
                    break

            if not readme_path:
                logger.warning("No README file found in repository: %s",
                             self.path)

            return readme_path
        except Exception as e:
            logger.error("Failed to get repository contents for %s: %s",
                       self.path, e)
            return None

    def to_json(self):
        """
        Convert repository data to JSON-serializable format

        This method creates a dictionary representation of the repository
        data that can be easily serialized to JSON.

        Returns:
            dict: Repository data as dictionary
        """
        logger.debug("Converting repository %s to JSON", self.path)
        attrs = self.repo.raw_data
        attrs["readme"] = self.readme
        logger.debug("JSON conversion completed for: %s", self.path)
        return attrs

    @property
    def filename(self):
        """
        Generate safe filename from repository name

        This property creates a filesystem-safe filename by replacing
        slashes with hyphens and converting to lowercase.

        Returns:
            str: Safe filename for the repository
        """
        filename = self.full_name.replace("/", "-").lower()
        logger.debug("Generated filename for %s: %s", self.path, filename)
        return filename

    @property
    def json_filename(self):
        """
        Generate JSON filename for repository data

        Returns:
            str: JSON filename with .json extension
        """
        json_filename = "%s.json" % self.filename
        logger.debug("Generated JSON filename for %s: %s", self.path,
                    json_filename)
        return json_filename

    @property
    def images(self):
        """
        Associated images for the repository

        Returns:
            list: List of image data associated with the repository
        """
        return self._images

    @images.setter
    def images(self, images):
        """
        Set associated images for the repository

        Args:
            images (list): List of image data to associate with repository
        """
        logger.debug("Setting images for repository %s: %d images",
                   self.path, len(images))
        self._images = images

    @property
    def created_days(self):
        """
        Calculate days since repository creation

        This property calculates the number of days between the repository
        creation date and the current date.

        Returns:
            int: Number of days since creation, 0 if cannot be calculated
        """
        try:
            date = datetime.datetime.strptime(
                self.created_at, '%Y-%m-%dT%H:%M:%SZ').date()
            days = (datetime.date.today() - date).days
            logger.debug("Repository %s created %d days ago",
                        self.path, days)
            return days
        except Exception as e:
            logger.error("Failed to calculate created days for %s: %s",
                       self.path, e)
            return 0
