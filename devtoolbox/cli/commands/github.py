"""
GitHub CLI commands for trending repositories and repository analysis
"""
import json
import logging
import typer
from typing import Optional

from devtoolbox.api_clients.github_client import GithubHandler

# Configure logging
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(help="GitHub related commands")


@app.command("trending")
def get_trending_repos(
    lang: str = typer.Option(
        "any",
        "-l", "--lang",
        help="Programming language filter (e.g. python, javascript, any)",
    ),
    lang_code: str = typer.Option(
        "any",
        "-c", "--lang-code",
        help="Spoken language filter (e.g. en, zh, any)",
    ),
    since: str = typer.Option(
        "today",
        "-s", "--since",
        help="Time range filter (today, weekly, monthly)",
    ),
    num: int = typer.Option(
        10,
        "-n", "--num",
        help="Maximum number of repositories to return",
    ),
    output: str = typer.Option(
        "table",
        "-o", "--output",
        help="Output format (table, json, csv)",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-f", "--save",
        help="Save results to file",
    ),
    debug: bool = typer.Option(
        False,
        "-d", "--debug",
        help="Enable debug logging",
    ),
):
    """
    Get trending repositories from GitHub
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.debug(
        "Getting trending repos with lang=%s, lang_code=%s, since=%s, "
        "num=%s, output=%s",
        lang, lang_code, since, num, output
    )

    try:
        # Initialize GitHub handler
        github_handler = GithubHandler()

        # Get trending repositories
        repos = github_handler.get_trendings(
            lang_code=lang_code,
            lang=lang,
            since=since,
            num=num
        )

        if output == "json":
            # Output as JSON
            result = []
            for repo in repos:
                repo_data = {
                    "name": repo.full_name,
                    "description": repo.description,
                    "url": repo.html_url,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "language": repo.language,
                    "main_languages": repo.main_languages,
                    "contributors_count": repo.contributors_count,
                    "created_days": repo.created_days,
                    "readme_length": len(repo.readme) if repo.readme else 0
                }
                result.append(repo_data)

            json_output = json.dumps(result, indent=2, ensure_ascii=False)
            typer.echo(json_output)

        elif output == "csv":
            # Output as CSV
            typer.echo("name,description,url,stars,forks,language,"
                      "main_languages,contributors_count,created_days,"
                      "readme_length")
            for repo in repos:
                main_langs = ";".join(repo.main_languages) if repo.main_languages else ""
                description = repo.description.replace(",", ";") if repo.description else ""
                typer.echo(f"{repo.full_name},{description},{repo.html_url},"
                          f"{repo.stargazers_count},{repo.forks_count},"
                          f"{repo.language},{main_langs},{repo.contributors_count},"
                          f"{repo.created_days},{len(repo.readme) if repo.readme else 0}")

        else:
            # Output as table
            typer.echo(f"\nFound {len(repos)} trending repositories:\n")
            typer.echo(f"{'Repository':<30} {'Language':<12} {'Stars':<8} "
                      f"{'Forks':<8} {'Contributors':<12} {'Created':<10}")
            typer.echo("-" * 90)

            for repo in repos:
                main_langs = ", ".join(repo.main_languages[:2]) if repo.main_languages else repo.language or "N/A"
                created_info = f"{repo.created_days}d ago" if repo.created_days else "N/A"
                typer.echo(f"{repo.full_name:<30} {main_langs:<12} "
                          f"{repo.stargazers_count:<8} {repo.forks_count:<8} "
                          f"{repo.contributors_count:<12} {created_info:<10}")

        # Save to file if requested
        if save:
            if output == "json":
                with open(save, 'w', encoding='utf-8') as f:
                    f.write(json_output)
            elif output == "csv":
                with open(save, 'w', encoding='utf-8') as f:
                    f.write("name,description,url,stars,forks,language,"
                           "main_languages,contributors_count,created_days,"
                           "readme_length\n")
                    for repo in repos:
                        main_langs = ";".join(repo.main_languages) if repo.main_languages else ""
                        description = repo.description.replace(",", ";") if repo.description else ""
                        f.write(f"{repo.full_name},{description},{repo.html_url},"
                               f"{repo.stargazers_count},{repo.forks_count},"
                               f"{repo.language},{main_langs},{repo.contributors_count},"
                               f"{repo.created_days},{len(repo.readme) if repo.readme else 0}\n")
            else:
                with open(save, 'w', encoding='utf-8') as f:
                    f.write(f"Found {len(repos)} trending repositories:\n\n")
                    f.write(f"{'Repository':<30} {'Language':<12} {'Stars':<8} "
                           f"{'Forks':<8} {'Contributors':<12} {'Created':<10}\n")
                    f.write("-" * 90 + "\n")
                    for repo in repos:
                        main_langs = ", ".join(repo.main_languages[:2]) if repo.main_languages else repo.language or "N/A"
                        created_info = f"{repo.created_days}d ago" if repo.created_days else "N/A"
                        f.write(f"{repo.full_name:<30} {main_langs:<12} "
                               f"{repo.stargazers_count:<8} {repo.forks_count:<8} "
                               f"{repo.contributors_count:<12} {created_info:<10}\n")

            typer.echo(f"\nResults saved to {save}")

    except Exception as e:
        logger.error("Failed to get trending repositories: %s", str(e), exc_info=True)
        typer.echo(f"Failed to get trending repositories: {str(e)}")
        raise typer.Exit(1)


@app.command("repo")
def get_repo_info(
    repo_path: str = typer.Argument(
        ...,
        help="Repository path in format 'owner/repo'",
    ),
    output: str = typer.Option(
        "table",
        "-o", "--output",
        help="Output format (table, json)",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-f", "--save",
        help="Save results to file",
    ),
    debug: bool = typer.Option(
        False,
        "-d", "--debug",
        help="Enable debug logging",
    ),
):
    """
    Get detailed information about a specific repository
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.debug("Getting repo info for: %s", repo_path)

    try:
        # Initialize GitHub handler
        github_handler = GithubHandler()

        # Get repository information
        repo = github_handler.github.get_repo(repo_path)

        if output == "json":
            # Output as JSON
            repo_data = {
                "name": repo.full_name,
                "description": repo.description,
                "url": repo.html_url,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "language": repo.language,
                "created_at": repo.created_at,
                "updated_at": repo.updated_at,
                "size": repo.size,
                "open_issues": repo.open_issues_count,
                "license": repo.license.name if repo.license else None,
                "topics": repo.get_topics(),
                "readme_length": len(repo.get_readme().decoded_content.decode('utf-8')) if repo.get_readme() else 0
            }

            json_output = json.dumps(repo_data, indent=2, ensure_ascii=False)
            typer.echo(json_output)

        else:
            # Output as table
            typer.echo(f"\nRepository: {repo.full_name}")
            typer.echo(f"Description: {repo.description or 'No description'}")
            typer.echo(f"URL: {repo.html_url}")
            typer.echo(f"Language: {repo.language or 'N/A'}")
            typer.echo(f"Stars: {repo.stargazers_count}")
            typer.echo(f"Forks: {repo.forks_count}")
            typer.echo(f"Open Issues: {repo.open_issues_count}")
            typer.echo(f"Size: {repo.size} KB")
            typer.echo(f"License: {repo.license.name if repo.license else 'N/A'}")
            typer.echo(f"Created: {repo.created_at}")
            typer.echo(f"Updated: {repo.updated_at}")

            topics = repo.get_topics()
            if topics:
                typer.echo(f"Topics: {', '.join(topics)}")

        # Save to file if requested
        if save:
            if output == "json":
                with open(save, 'w', encoding='utf-8') as f:
                    f.write(json_output)
            else:
                with open(save, 'w', encoding='utf-8') as f:
                    f.write(f"Repository: {repo.full_name}\n")
                    f.write(f"Description: {repo.description or 'No description'}\n")
                    f.write(f"URL: {repo.html_url}\n")
                    f.write(f"Language: {repo.language or 'N/A'}\n")
                    f.write(f"Stars: {repo.stargazers_count}\n")
                    f.write(f"Forks: {repo.forks_count}\n")
                    f.write(f"Open Issues: {repo.open_issues_count}\n")
                    f.write(f"Size: {repo.size} KB\n")
                    f.write(f"License: {repo.license.name if repo.license else 'N/A'}\n")
                    f.write(f"Created: {repo.created_at}\n")
                    f.write(f"Updated: {repo.updated_at}\n")
                    topics = repo.get_topics()
                    if topics:
                        f.write(f"Topics: {', '.join(topics)}\n")

            typer.echo(f"\nResults saved to {save}")

    except Exception as e:
        logger.error("Failed to get repository info: %s", str(e), exc_info=True)
        typer.echo(f"Failed to get repository info: {str(e)}")
        raise typer.Exit(1)


@app.command("setup")
def setup_github_token(
    token: str = typer.Option(
        ...,
        "-t", "--token",
        help="GitHub personal access token",
    ),
):
    """
    Set up GitHub authentication token
    """
    try:
        # Test the token by creating a handler
        github_handler = GithubHandler(token=token)

        # Try to get user info to verify token
        user = github_handler.github.get_user()
        typer.echo(f"✅ GitHub token is valid! Authenticated as: {user.login}")
        typer.echo(f"Token will be used from environment variable GITHUB_TOKEN")
        typer.echo(f"To set it permanently, add to your shell profile:")
        typer.echo(f"export GITHUB_TOKEN='{token}'")

    except Exception as e:
        logger.error("Failed to validate GitHub token: %s", str(e))
        typer.echo(f"❌ Invalid GitHub token: {str(e)}")
        raise typer.Exit(1)