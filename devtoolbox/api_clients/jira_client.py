import os
import logging
import re
from typing import List, Optional, Dict, Union, Literal, Any
from jira import JIRA
from datetime import datetime


class JiraClient:
    """A wrapper class for JIRA operations commonly used in the project.

    This class provides a convenient interface for interacting with JIRA's
    API. It allows users to perform various operations such as searching
    for issues, updating issue labels, and managing epics and sprints.

    Usage:
        1. Initialize the JiraClient with your JIRA credentials:
            jira_client = JiraClient(
                jira_url="https://your_jira_instance.com",
                username="your_username",
                password="your_password"
            )

        2. Use the methods provided to interact with JIRA:
            - Search for issues using JQL:
                issues = jira_client.search_issues(
                    "project = YOUR_PROJECT_KEY"
                )

            - Get epics in a project:
                epics = jira_client.get_epics(
                    "YOUR_PROJECT_KEY", "epic_summary"
                )

            - Update labels for an issue:
                jira_client.update_issue_labels(
                    "ISSUE_KEY", ["label1", "label2"]
                )

            - Get active sprints for a project:
                active_sprints = jira_client.get_active_sprints(
                    "YOUR_PROJECT_KEY"
                )

    Attributes:
        EPIC_LINK_FIELD (str): The custom field ID for the epic link.
        EPIC_NAME_FIELD (str): The custom field ID for the epic name.
        SPRINT_FIELD_NAME (str): The name of the sprint field.
    """

    # Common custom fields
    EPIC_LINK_FIELD = "customfield_10101"
    EPIC_NAME_FIELD = "customfield_10103"
    SPRINT_FIELD_NAME = "Sprint"

    def __init__(self, jira_url: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 *args, **kwargs):
        """Initialize JIRA client with environment variables or
        provided parameters."""
        self.jira_url = jira_url or os.environ.get("JIRA_URL", "")
        self.username = username or os.environ.get("JIRA_USERNAME", "")
        self.password = password or os.environ.get("JIRA_PASSWORD", "")

        if not self.jira_url:
            raise ValueError("Please provide JIRA_URL either through the "
                             "initialization function or as an environment "
                             "variable.")
        if not self.username or not self.password:
            raise ValueError("Please provide JIRA_USERNAME and "
                             "JIRA_PASSWORD either through the "
                             "initialization function or as environment "
                             "variables.")

        try:
            self.client = JIRA(
                server=self.jira_url,
                basic_auth=(self.username, self.password),
                *args, **kwargs
            )
            # Test the connection by making a simple request
            self.client.myself()
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                raise ValueError(
                    "Authentication failed. Please check your JIRA "
                    "username and password."
                )
            elif "403" in error_msg:
                raise ValueError(
                    "Access forbidden. Please check if your account "
                    "has the necessary permissions."
                )
            elif "404" in error_msg:
                raise ValueError(
                    f"JIRA server not found at {self.jira_url}. "
                    "Please check the URL."
                )
            else:
                raise ValueError(
                    f"Failed to connect to JIRA: {error_msg}"
                )

    def search_issues(self, jql: str, max_results: int = None) -> List:
        """Search issues using JQL query.

        Args:
            jql: The JQL query string to search with.
            max_results: Maximum number of results to return. If None,
            returns all results.

        Returns:
            List of issues matching the JQL query.
        """
        return self.client.search_issues(
            jql, maxResults=False if max_results is None else max_results
        )

    def get_issue_details(
        self,
        issue_key: str,
        output_format: Literal["json", "markdown"] = "json"
    ) -> Union[Dict, str]:
        """Get detailed information about a JIRA issue including comments
        and history.

        Args:
            issue_key: The JIRA issue key (e.g., 'PROJ-123')
            output_format: The desired output format ('json' or 'markdown')

        Returns:
            Either a dictionary (for JSON) or a string (for markdown)
            containing the issue details, comments, and history.
        """
        try:
            # Get the issue with all fields and expanded changelog
            issue = self.client.issue(
                issue_key,
                expand='changelog,comments'
            )

            # Collect basic issue information
            issue_data = {
                'key': issue.key,
                'summary': issue.fields.summary,
                'status': issue.fields.status.name,
                'issue_type': issue.fields.issuetype.name,
                'priority': (issue.fields.priority.name
                             if issue.fields.priority else None),
                'assignee': (issue.fields.assignee.displayName
                             if issue.fields.assignee else None),
                'reporter': (issue.fields.reporter.displayName
                             if issue.fields.reporter else None),
                'created': issue.fields.created,
                'updated': issue.fields.updated,
                'description': issue.fields.description or '',
                'labels': issue.fields.labels,
                'components': [c.name if hasattr(c, 'name') else c
                             for c in issue.fields.components],
                'fix_versions': [v.name if hasattr(v, 'name') else v
                               for v in issue.fields.fixVersions],
                'epic_link': getattr(issue.fields, self.EPIC_LINK_FIELD, None),
            }

            # Collect comments
            comments = []
            for comment in issue.fields.comment.comments:
                comments.append({
                    'author': comment.author.displayName,
                    'created': comment.created,
                    'body': comment.body,
                    'updated': comment.updated
                })
            issue_data['comments'] = comments

            # Collect history
            history = []
            for history_item in issue.changelog.histories:
                changes = []
                for item in history_item.items:
                    changes.append({
                        'field': item.field,
                        'from': item.fromString,
                        'to': item.toString
                    })
                history.append({
                    'author': history_item.author.displayName,
                    'created': history_item.created,
                    'changes': changes
                })
            issue_data['history'] = history

            if output_format == "json":
                return issue_data
            else:
                return self._format_issue_as_markdown(issue_data)

        except Exception as e:
            logging.error(f"Error fetching issue details for {issue_key}: "
                          f"{str(e)}")
            raise

    def _format_issue_as_markdown(self, issue_data: Dict) -> str:
        """Convert issue data to markdown format.

        Args:
            issue_data: Dictionary containing issue information

        Returns:
            Formatted markdown string
        """
        def convert_jira_content(content: str) -> str:
            """Convert JIRA content to Markdown, including image handling.

            Converts:
            - JIRA image: !image-name.png|thumbnail!
            - JIRA attachment: [^attachment-name.pdf]
            - Plain URLs: http(s)://example.com
            to standard Markdown format.
            """
            if not content:
                return ""

            # Convert JIRA image syntax to Markdown
            image_pattern = r'!([^|!]+)(?:\|[^!]+)?!'
            content = re.sub(
                image_pattern,
                lambda m: (f"![{m.group(1)}]({self.jira_url}/secure/"
                          f"attachment/{issue_data['key']}/{m.group(1)})"),
                content
            )

            # Convert JIRA attachment syntax to Markdown
            attachment_pattern = r'\[\^([^\]]+)\]'
            content = re.sub(
                attachment_pattern,
                lambda m: (f"[{m.group(1)}]({self.jira_url}/secure/"
                          f"attachment/{issue_data['key']}/{m.group(1)})"),
                content
            )

            # Convert plain URLs to Markdown links
            url_pattern = r'(?<![\[\(])(https?://[^\s\)<>]+)(?![^\[]*\])'
            content = re.sub(
                url_pattern,
                lambda m: f"[{m.group(1)}]({m.group(1)})",
                content
            )

            return content

        # Format basic information
        md = (f"# {issue_data['key']}: "
              f"{convert_jira_content(issue_data['summary'])}\n\n")

        md += "## Basic Information\n"
        md += f"- **Type:** {issue_data['issue_type']}\n"
        md += f"- **Status:** {issue_data['status']}\n"
        md += f"- **Priority:** {issue_data['priority']}\n"
        md += f"- **Assignee:** {issue_data['assignee']}\n"
        md += f"- **Reporter:** {issue_data['reporter']}\n"
        md += f"- **Created:** {issue_data['created']}\n"
        md += f"- **Updated:** {issue_data['updated']}\n"

        if issue_data['labels']:
            md += f"- **Labels:** {', '.join(issue_data['labels'])}\n"
        if issue_data['components']:
            md += f"- **Components:** {', '.join(issue_data['components'])}\n"
        if issue_data['fix_versions']:
            md += f"- **Fix Versions:** {', '.join(issue_data['fix_versions'])}\n"
        if issue_data['epic_link']:
            epic_link = str(issue_data['epic_link'])
            md += f"- **Epic Link:** {convert_jira_content(epic_link)}\n"

        # Description with content conversion
        if issue_data['description']:
            md += "\n## Description\n"
            md += convert_jira_content(issue_data['description']) + "\n"

        # Comments with content conversion
        if issue_data['comments']:
            md += "\n## Comments\n"
            for comment in issue_data['comments']:
                md += (f"### {comment['author']} - {comment['created']}\n"
                       f"{convert_jira_content(comment['body'])}\n\n")

        # History with content conversion
        if issue_data['history']:
            md += "\n## History\n"
            for history in issue_data['history']:
                md += f"### {history['author']} - {history['created']}\n"
                for change in history['changes']:
                    # Convert both 'from' and 'to' values
                    from_value = (convert_jira_content(str(change['from']))
                                if change['from'] else '')
                    to_value = (convert_jira_content(str(change['to']))
                              if change['to'] else '')
                    md += (f"- Changed **{change['field']}** from "
                           f"'{from_value}' to '{to_value}'\n")
                md += "\n"

        return md

    def update_issue_labels(self, issue_key: str, labels: List[str]) -> None:
        """Update labels for a specific issue.

        This method checks if the labels already exist before updating.
        If all labels are already present, it skips the update operation.

        Args:
            issue_key: The JIRA issue key (e.g., 'PROJ-123')
            labels: List of labels to add to the issue

        Example:
            jira_client.update_issue_labels('PROJ-123', ['label1', 'label2'])
        """
        try:
            issue = self.client.issue(issue_key)
            existing_labels = issue.fields.labels

            # Convert lists to sets for comparison
            new_labels_set = set(labels)
            existing_labels_set = set(existing_labels)

            # Check if all new labels are already present
            if new_labels_set.issubset(existing_labels_set):
                logging.warning(
                    f"Skip adding existing labels {labels} to {issue_key}"
                )
                return

            # Add new labels to existing ones
            updated_labels = list(existing_labels_set.union(new_labels_set))
            issue.update(fields={"labels": updated_labels})

            logging.info(
                f"Labels updated for issue {issue_key}: "
                f"added {list(new_labels_set - existing_labels_set)}"
            )

        except Exception as e:
            logging.error(
                f"Error updating labels for {issue_key}: {str(e)}"
            )
            raise

    def get_active_sprints(self, project_key: str) -> List:
        """Get all active sprints for a project."""
        active_sprints = []
        boards = self.client.boards(projectKeyOrID=project_key)

        for board in boards:
            if board.type == "scrum":
                sprints = self.client.sprints(board.id)
                for sprint in sprints:
                    if (sprint.state == "active" and
                            sprint not in active_sprints):
                        active_sprints.append(sprint)

        return active_sprints

    def get_project_versions(self, project_key: str) -> List:
        """Get all versions for a project."""
        return self.client.project_versions(project_key)

    def _get_sprint_field(self) -> Optional[str]:
        """Get the sprint field ID."""
        fields = self.client.fields()
        for field in fields:
            if field["name"] == self.SPRINT_FIELD_NAME:
                return field["id"]
        return None

    def _prepare_field_values(
        self,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        issue_type: Optional[str] = None,
        assignee: Optional[str] = None,
        priority: Optional[str] = None,
        labels: Optional[List[str]] = None,
        components: Optional[List[str]] = None,
        fix_versions: Optional[List[str]] = None,
        epic_link: Optional[str] = None,
        epic_name: Optional[str] = None,
        sprint: Optional[str] = None,
        **kwargs
    ) -> dict:
        """Prepare field values for issue creation or update.

        Args:
            summary: Issue summary/title
            description: Issue description
            issue_type: Issue type name (e.g., 'Story', 'Bug', 'Task')
            assignee: Username or account ID of the assignee
            priority: Priority name (e.g., 'High', 'Medium', 'Low')
            labels: List of labels to apply to the issue
            components: List of component names to associate with the issue
            fix_versions: List of fix version names to associate with the issue
            epic_link: Key of the epic to link this issue to (e.g., 'PROJ-123')
            epic_name: Name of the epic (only valid for epic issue types)
            sprint: Sprint name or state ('active', 'next', 'backlog')
            **kwargs: Additional JIRA fields to set (using field names as keys)

        Returns:
            dict: Prepared field values ready for JIRA API

        Raises:
            ValueError: If epic_name is set for non-epic issue types
        """
        fields = {}

        # Handle standard fields
        if summary is not None:
            fields['summary'] = summary
        if description is not None:
            fields['description'] = description
        if issue_type is not None:
            fields['issuetype'] = {'name': issue_type}
        if assignee is not None:
            fields['assignee'] = {'name': assignee}
        if priority is not None:
            fields['priority'] = {'name': priority}
        if labels is not None:
            fields['labels'] = labels

        # Handle components
        if components is not None:
            fields['components'] = [{'name': component} for component in components]

        # Handle fix versions
        if fix_versions is not None:
            fields['fixVersions'] = [{'name': version} for version in fix_versions]

        # Handle epic fields
        if epic_link is not None:
            fields[self.EPIC_LINK_FIELD] = epic_link
        if epic_name is not None:
            if issue_type and issue_type.lower() != 'epic':
                raise ValueError("Epic name can only be set for epic issue types")
            fields[self.EPIC_NAME_FIELD] = epic_name

        # Add any additional fields from kwargs
        fields.update(kwargs)

        return fields

    def create_issue(
        self,
        project_key: str,
        summary: str,
        issue_type: str = 'Story',
        description: Optional[str] = None,
        assignee: Optional[str] = None,
        priority: Optional[str] = None,
        labels: Optional[List[str]] = None,
        components: Optional[List[str]] = None,
        fix_versions: Optional[List[str]] = None,
        epic_link: Optional[str] = None,
        epic_name: Optional[str] = None,
        sprint: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create a new JIRA issue.

        Args:
            project_key (str): The project key (e.g., 'PROJ')
            summary (str): Issue summary/title
            issue_type (str, optional): Issue type name. Defaults to 'Story'
            description (str, optional): Issue description
            assignee (str, optional): Username of the assignee
            priority (str, optional): Priority name (e.g., 'High', 'Medium')
            labels (List[str], optional): List of labels
            components (List[str], optional): List of component names
            fix_versions (List[str], optional): List of fix version names
            epic_link (str, optional): Key of the epic to link this issue to
            epic_name (str, optional): Name of the epic (only for epic issue types)
            sprint (str, optional): Sprint name or state ('active', 'next', 'backlog')
            **kwargs: Additional fields to set

        Returns:
            str: The key of the created issue

        Raises:
            ValueError: If required fields are missing or invalid
            Exception: If there's an error creating the issue
        """
        try:
            # Prepare basic fields
            fields = self._prepare_field_values(
                summary=summary,
                description=description,
                issue_type=issue_type,
                assignee=assignee,
                priority=priority,
                labels=labels,
                components=components,
                fix_versions=fix_versions,
                epic_link=epic_link,
                epic_name=epic_name,
                **kwargs
            )

            # Add project
            fields['project'] = {'key': project_key}

            # Create the issue
            new_issue = self.client.create_issue(fields=fields)
            issue_key = new_issue.key
            logging.info(f"Created issue: {issue_key}")

            # Handle sprint assignment if specified
            if sprint is not None:
                self._update_sprint(issue_key, sprint)

            return issue_key

        except Exception as e:
            logging.error(f"Error creating issue: {str(e)}")
            raise

    def update_issue(
        self,
        issue_key: str,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        assignee: Optional[str] = None,
        priority: Optional[str] = None,
        status: Optional[str] = None,
        labels: Optional[List[str]] = None,
        components: Optional[List[str]] = None,
        fix_versions: Optional[List[str]] = None,
        epic_link: Optional[str] = None,
        epic_name: Optional[str] = None,
        sprint: Optional[str] = None,
        **kwargs
    ) -> None:
        """Update a JIRA issue with the specified fields."""
        try:
            issue = self.client.issue(issue_key)

            # Prepare fields for update
            update_fields = self._prepare_field_values(
                summary=summary,
                description=description,
                assignee=assignee,
                priority=priority,
                labels=labels,
                components=components,
                fix_versions=fix_versions,
                epic_link=epic_link,
                epic_name=epic_name,
                **kwargs
            )

            # Handle sprint update if specified
            if sprint is not None:
                self._update_sprint(issue_key, sprint)

            # Handle status transition if needed
            if status is not None:
                self._update_status(issue, status)

            # Update the issue if there are fields to update
            if update_fields:
                issue.update(fields=update_fields)
                logging.info(
                    f"Updated issue {issue_key} with fields: "
                    f"{list(update_fields.keys())}"
                )
            else:
                logging.info(f"No fields to update for issue {issue_key}")

        except Exception as e:
            logging.error(f"Error updating issue {issue_key}: {str(e)}")
            raise

    def _update_sprint(self, issue_key: str, sprint: str) -> None:
        """Update the sprint for an issue.

        Args:
            issue_key: The JIRA issue key (e.g., 'PROJ-123')
            sprint: Sprint name or state. Can be one of:
                   - Specific sprint name (e.g., "Sprint 1")
                   - "active" - Move to current active sprint
                   - "next" - Move to next planned sprint
                   - "backlog" - Remove from sprint/move to backlog

        Raises:
            ValueError: If sprint field is not found or board cannot be determined
            Exception: If there's an error during sprint update
        """
        try:
            sprint_field = self._get_sprint_field()
            if not sprint_field:
                raise ValueError("Sprint field not found in JIRA configuration")

            board_id = self._get_board_for_issue(issue_key)
            if not board_id:
                raise ValueError(f"Could not find board for issue {issue_key}")

            sprint_id = None
            if sprint.lower() == 'active':
                active_sprints = self.client.sprints(board_id, state='active')
                if active_sprints:
                    sprint_id = active_sprints[0].id
            elif sprint.lower() == 'next':
                future_sprints = self.client.sprints(board_id, state='future')
                if future_sprints:
                    sprint_id = future_sprints[0].id
            elif sprint.lower() == 'backlog':
                sprint_id = None
            else:
                all_sprints = self.client.sprints(board_id)
                matching_sprint = next(
                    (s for s in all_sprints if s.name == sprint),
                    None
                )
                if matching_sprint:
                    sprint_id = matching_sprint.id

            if sprint_id is not None or sprint.lower() == 'backlog':
                self.client.issue(issue_key).update(
                    fields={sprint_field: sprint_id}
                )
                logging.info(f"Moved issue {issue_key} to sprint: {sprint}")
            else:
                logging.warning(
                    f"Could not find sprint '{sprint}' for issue {issue_key}"
                )

        except Exception as e:
            logging.error(
                f"Error updating sprint for issue {issue_key}: {str(e)}"
            )
            raise

    def _update_status(self, issue: Any, status: str) -> None:
        """Update the status of an issue.

        Args:
            issue: JIRA issue object
            status: Target status name (case-insensitive)

        Note:
            The status transition must be valid according to the workflow.
            If the transition is not available, a warning will be logged.
        """
        available_transitions = self.client.transitions(issue)
        transition_id = next(
            (t['id'] for t in available_transitions
             if t['to']['name'].lower() == status.lower()),
            None
        )
        if transition_id:
            self.client.transition_issue(issue, transition_id)
            logging.info(f"Updated status to {status}")
        else:
            logging.warning(f"Could not find transition to status '{status}'")

    def _get_board_for_issue(self, issue_key: str) -> Optional[int]:
        """Get the board ID for a given issue.

        Args:
            issue_key: The JIRA issue key (e.g., 'PROJ-123')

        Returns:
            Optional[int]: The board ID if found, None if no board exists
                          or an error occurs

        Note:
            Returns the first board found for the project, which is typically
            the main board. For projects with multiple boards, this may need
            to be enhanced.
        """
        try:
            # Get project key from issue key
            project_key = issue_key.split('-')[0]

            # Search for boards in the project
            boards = self.client.boards(projectKeyOrID=project_key)

            # Return the first board ID found (usually the main board)
            if boards:
                return boards[0].id
            return None
        except Exception as e:
            logging.error(f"Error finding board for issue {issue_key}: {str(e)}")
            return None

    def delete_issue(
        self,
        issue_key: str,
        verify: bool = True,
        subtasks: bool = True
    ) -> bool:
        """Delete a JIRA issue.

        Args:
            issue_key (str): The JIRA issue key to delete (e.g., 'PROJ-123')
            verify (bool, optional): If True, verify the issue exists and is
                                   deletable before attempting deletion.
                                   Defaults to True.
            subtasks (bool, optional): If True, delete subtasks before deleting
                                     the parent issue. Defaults to True.

        Returns:
            bool: True if deletion was successful, False otherwise

        Raises:
            ValueError: If the issue doesn't exist or can't be deleted
            Exception: If there's an error during deletion
        """
        try:
            if not verify:
                # Direct deletion without verification
                self.client.issue(issue_key).delete()
                logging.info(f"Successfully deleted issue: {issue_key}")
                return True

            # Verify issue exists and get its details
            issue = self.client.issue(issue_key)

            # Check if issue is a subtask
            if (hasattr(issue.fields, 'parent') and
                    issue.fields.parent is not None):
                logging.info(
                    f"Deleting subtask {issue_key} of "
                    f"{issue.fields.parent.key}"
                )
                issue.delete()
                return True

            # Check if issue has subtasks
            if hasattr(issue.fields, 'subtasks') and issue.fields.subtasks:
                if not subtasks:
                    raise ValueError(
                        f"Issue {issue_key} has subtasks and subtasks=False. "
                        "Please delete subtasks first or set subtasks=True"
                    )
                # Delete subtasks first
                for subtask in issue.fields.subtasks:
                    subtask_key = subtask.key
                    logging.info(
                        f"Deleting subtask {subtask_key} of {issue_key}"
                    )
                    if not self.delete_issue(subtask_key, verify=True):
                        logging.error(
                            f"Failed to delete subtask {subtask_key}"
                        )
                        return False

            # Delete the main issue
            issue.delete()
            logging.info(f"Successfully deleted issue: {issue_key}")
            return True

        except Exception as e:
            error_msg = str(e)
            if any(msg in error_msg for msg in
                   ["Issue Does Not Exist", "Not Found", "does not exist"]):
                logging.warning(f"Issue {issue_key} does not exist")
                return False
            else:
                logging.error(
                    f"Error deleting issue {issue_key}: {error_msg}"
                )
                raise
