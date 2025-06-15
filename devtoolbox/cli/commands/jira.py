"""
JIRA related commands
"""
import typer
import logging
from typing import Optional, List
from devtoolbox.api_clients.jira_client import JiraClient
from devtoolbox.cli.utils import setup_logging


# Configure logging
logger = logging.getLogger("devtoolbox.jira")
app = typer.Typer(help="JIRA related commands")


@app.callback()
def callback(
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug mode",
    ),
):
    """
    JIRA command line tool
    """
    global logger
    logger = setup_logging(debug, "devtoolbox.jira")


@app.command("search")
def search_issues(
    jql: str = typer.Argument(
        ...,
        help="JQL query string",
    ),
    max_results: Optional[int] = typer.Option(
        None,
        "-m", "--max-results",
        help="Maximum number of results to return",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    Search issues using JQL query
    """
    logger.debug(
        "Searching issues with JQL: %s (max_results=%s)",
        jql, max_results
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        issues = client.search_issues(jql, max_results)
        for issue in issues:
            typer.echo(f"{issue.key}: {issue.fields.summary}")
    except Exception as e:
        logger.error(
            "Failed to search issues: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to search issues: {str(e)}")
        raise typer.Exit(1)


@app.command("get")
def get_issue(
    issue_key: str = typer.Argument(
        ...,
        help="JIRA issue key (e.g., 'PROJ-123')",
    ),
    format: str = typer.Option(
        "json",
        "-f", "--format",
        help="Output format (json or markdown)",
        case_sensitive=False,
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    Get detailed information about a JIRA issue
    """
    logger.debug(
        "Getting issue details for %s (format=%s)",
        issue_key, format
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        result = client.get_issue_details(issue_key, format)
        typer.echo(result)
    except Exception as e:
        logger.error(
            "Failed to get issue details: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to get issue details: {str(e)}")
        raise typer.Exit(1)


@app.command("create")
def create_issue(
    project_key: str = typer.Argument(
        ...,
        help="Project key",
    ),
    summary: str = typer.Argument(
        ...,
        help="Issue summary",
    ),
    issue_type: str = typer.Option(
        "Story",
        "-t", "--type",
        help="Issue type",
    ),
    description: Optional[str] = typer.Option(
        None,
        "-d", "--description",
        help="Issue description",
    ),
    assignee: Optional[str] = typer.Option(
        None,
        "-a", "--assignee",
        help="Assignee username",
    ),
    priority: Optional[str] = typer.Option(
        None,
        "-P", "--priority",
        help="Issue priority",
    ),
    labels: Optional[List[str]] = typer.Option(
        None,
        "-l", "--label",
        help="Issue labels (can be used multiple times)",
    ),
    components: Optional[List[str]] = typer.Option(
        None,
        "-c", "--component",
        help="Issue components (can be used multiple times)",
    ),
    fix_versions: Optional[List[str]] = typer.Option(
        None,
        "-v", "--fix-version",
        help="Fix versions (can be used multiple times)",
    ),
    epic_link: Optional[str] = typer.Option(
        None,
        "-e", "--epic-link",
        help="Epic link",
    ),
    epic_name: Optional[str] = typer.Option(
        None,
        "-E", "--epic-name",
        help="Epic name",
    ),
    sprint: Optional[str] = typer.Option(
        None,
        "-s", "--sprint",
        help="Sprint name",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    Create a new JIRA issue
    """
    logger.debug(
        "Creating issue in project %s: %s (type=%s)",
        project_key, summary, issue_type
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        issue_key = client.create_issue(
            project_key=project_key,
            summary=summary,
            issue_type=issue_type,
            description=description,
            assignee=assignee,
            priority=priority,
            labels=labels,
            components=components,
            fix_versions=fix_versions,
            epic_link=epic_link,
            epic_name=epic_name,
            sprint=sprint
        )
        typer.echo(f"Successfully created issue: {issue_key}")
    except Exception as e:
        logger.error(
            "Failed to create issue: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to create issue: {str(e)}")
        raise typer.Exit(1)


@app.command("update")
def update_issue(
    issue_key: str = typer.Argument(
        ...,
        help="JIRA issue key (e.g., 'PROJ-123')",
    ),
    summary: Optional[str] = typer.Option(
        None,
        "-s", "--summary",
        help="Issue summary",
    ),
    description: Optional[str] = typer.Option(
        None,
        "-d", "--description",
        help="Issue description",
    ),
    assignee: Optional[str] = typer.Option(
        None,
        "-a", "--assignee",
        help="Assignee username",
    ),
    priority: Optional[str] = typer.Option(
        None,
        "-P", "--priority",
        help="Issue priority",
    ),
    status: Optional[str] = typer.Option(
        None,
        "-S", "--status",
        help="Issue status",
    ),
    labels: Optional[List[str]] = typer.Option(
        None,
        "-l", "--label",
        help="Issue labels (can be used multiple times)",
    ),
    components: Optional[List[str]] = typer.Option(
        None,
        "-c", "--component",
        help="Issue components (can be used multiple times)",
    ),
    fix_versions: Optional[List[str]] = typer.Option(
        None,
        "-v", "--fix-version",
        help="Fix versions (can be used multiple times)",
    ),
    epic_link: Optional[str] = typer.Option(
        None,
        "-e", "--epic-link",
        help="Epic link",
    ),
    epic_name: Optional[str] = typer.Option(
        None,
        "-E", "--epic-name",
        help="Epic name",
    ),
    sprint: Optional[str] = typer.Option(
        None,
        "-s", "--sprint",
        help="Sprint name",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    Update an existing JIRA issue
    """
    logger.debug(
        "Updating issue %s",
        issue_key
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        client.update_issue(
            issue_key=issue_key,
            summary=summary,
            description=description,
            assignee=assignee,
            priority=priority,
            status=status,
            labels=labels,
            components=components,
            fix_versions=fix_versions,
            epic_link=epic_link,
            epic_name=epic_name,
            sprint=sprint
        )
        typer.echo(f"Successfully updated issue: {issue_key}")
    except Exception as e:
        logger.error(
            "Failed to update issue: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to update issue: {str(e)}")
        raise typer.Exit(1)


@app.command("delete")
def delete_issue(
    issue_key: str = typer.Argument(
        ...,
        help="JIRA issue key (e.g., 'PROJ-123')",
    ),
    verify: bool = typer.Option(
        True,
        "-v", "--verify",
        help="Verify deletion",
    ),
    subtasks: bool = typer.Option(
        True,
        "-s", "--subtasks",
        help="Delete subtasks",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    Delete a JIRA issue
    """
    logger.debug(
        "Deleting issue %s (verify=%s, subtasks=%s)",
        issue_key, verify, subtasks
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        if client.delete_issue(issue_key, verify, subtasks):
            typer.echo(f"Successfully deleted issue: {issue_key}")
        else:
            typer.echo(f"Failed to delete issue: {issue_key}")
    except Exception as e:
        logger.error(
            "Failed to delete issue: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to delete issue: {str(e)}")
        raise typer.Exit(1)


@app.command("sprints")
def list_sprints(
    project_key: str = typer.Argument(
        ...,
        help="Project key",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    List active sprints for a project
    """
    logger.debug(
        "Listing active sprints for project %s",
        project_key
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        sprints = client.get_active_sprints(project_key)
        for sprint in sprints:
            typer.echo(f"{sprint.name}: {sprint.state}")
    except Exception as e:
        logger.error(
            "Failed to list sprints: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to list sprints: {str(e)}")
        raise typer.Exit(1)


@app.command("versions")
def list_versions(
    project_key: str = typer.Argument(
        ...,
        help="Project key",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    List versions for a project
    """
    logger.debug(
        "Listing versions for project %s",
        project_key
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        versions = client.get_project_versions(project_key)
        for version in versions:
            typer.echo(f"{version.name}: {version.released}")
    except Exception as e:
        logger.error(
            "Failed to list versions: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to list versions: {str(e)}")
        raise typer.Exit(1)


@app.command("comment")
def add_comment(
    issue_key: str = typer.Argument(
        ...,
        help="JIRA issue key (e.g., 'PROJ-123')",
    ),
    comment: str = typer.Argument(
        ...,
        help="Comment content",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    Add a comment to a JIRA issue
    """
    logger.debug(
        "Adding comment to issue %s",
        issue_key
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        client.client.add_comment(issue_key, comment)
        typer.echo(f"Successfully added comment to issue: {issue_key}")
    except Exception as e:
        logger.error(
            "Failed to add comment: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to add comment: {str(e)}")
        raise typer.Exit(1)


@app.command("comments")
def list_comments(
    issue_key: str = typer.Argument(
        ...,
        help="JIRA issue key (e.g., 'PROJ-123')",
    ),
    format: str = typer.Option(
        "text",
        "-f", "--format",
        help="Output format (text or json)",
        case_sensitive=False,
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    List comments for a JIRA issue
    """
    logger.debug(
        "Listing comments for issue %s (format=%s)",
        issue_key, format
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        issue = client.client.issue(issue_key, expand='comments')
        comments = issue.fields.comment.comments

        if format.lower() == "json":
            import json
            result = []
            for comment in comments:
                result.append({
                    'author': comment.author.displayName,
                    'created': comment.created,
                    'body': comment.body,
                    'updated': comment.updated
                })
            typer.echo(json.dumps(result, indent=2))
        else:
            for comment in comments:
                typer.echo(
                    f"[{comment.created}] {comment.author.displayName}:"
                )
                typer.echo(comment.body)
                typer.echo("---")
    except Exception as e:
        logger.error(
            "Failed to list comments: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to list comments: {str(e)}")
        raise typer.Exit(1)


@app.command("delete-comment")
def delete_comment(
    issue_key: str = typer.Argument(
        ...,
        help="JIRA issue key (e.g., 'PROJ-123')",
    ),
    comment_id: str = typer.Argument(
        ...,
        help="Comment ID",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    Delete a comment from a JIRA issue
    """
    logger.debug(
        "Deleting comment %s from issue %s",
        comment_id, issue_key
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        client.client.delete_comment(issue_key, comment_id)
        typer.echo(
            f"Successfully deleted comment {comment_id} from issue: {issue_key}"
        )
    except Exception as e:
        logger.error(
            "Failed to delete comment: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to delete comment: {str(e)}")
        raise typer.Exit(1)


@app.command("update-comment")
def update_comment(
    issue_key: str = typer.Argument(
        ...,
        help="JIRA issue key (e.g., 'PROJ-123')",
    ),
    comment_id: str = typer.Argument(
        ...,
        help="Comment ID",
    ),
    comment: str = typer.Argument(
        ...,
        help="New comment content",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u", "--url",
        help="JIRA URL",
    ),
    username: Optional[str] = typer.Option(
        None,
        "-n", "--username",
        help="JIRA username",
    ),
    password: Optional[str] = typer.Option(
        None,
        "-p", "--password",
        help="JIRA password",
    ),
):
    """
    Update a comment on a JIRA issue
    """
    logger.debug(
        "Updating comment %s on issue %s",
        comment_id, issue_key
    )

    try:
        client = JiraClient(
            jira_url=url,
            username=username,
            password=password
        )
        client.client.update_comment(issue_key, comment_id, comment)
        typer.echo(
            f"Successfully updated comment {comment_id} on issue: {issue_key}"
        )
    except Exception as e:
        logger.error(
            "Failed to update comment: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to update comment: {str(e)}")
        raise typer.Exit(1)