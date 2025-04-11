import unittest
import os
from unittest.mock import patch, MagicMock

from devtoolbox.jira_client import JiraClient
from tests.utils.test_logging import setup_test_logging

# Initialize logging
logger = setup_test_logging()


class TestJiraClient(unittest.TestCase):
    """Test cases for JiraClient class"""

    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test fixtures")
        # Set up test environment variables
        os.environ["JIRA_URL"] = "https://test.atlassian.net"
        os.environ["JIRA_USERNAME"] = "test_user"
        os.environ["JIRA_PASSWORD"] = "test_password"

        # Create mock JIRA client
        self.mock_jira = MagicMock()
        self.mock_jira.search_issues = MagicMock()
        self.mock_jira.issue = MagicMock()
        self.mock_jira.create_issue = MagicMock()
        self.mock_jira.transitions = MagicMock()
        self.mock_jira.transition_issue = MagicMock()
        self.mock_jira.boards = MagicMock()
        self.mock_jira.sprints = MagicMock()
        self.mock_jira.project_versions = MagicMock()
        self.mock_jira.fields = MagicMock()

        # Create test data
        self.test_issue_key = "TEST-123"
        self.test_project_key = "TEST"
        self.test_issue = MagicMock()
        self.test_issue.key = self.test_issue_key
        self.test_issue.fields = MagicMock()
        self.test_issue.fields.summary = "Test Issue"
        self.test_issue.fields.description = "Test Description"
        self.test_issue.fields.status = MagicMock()
        self.test_issue.fields.status.name = "To Do"
        self.test_issue.fields.issuetype = MagicMock()
        self.test_issue.fields.issuetype.name = "Story"
        self.test_issue.fields.priority = MagicMock()
        self.test_issue.fields.priority.name = "High"
        self.test_issue.fields.assignee = MagicMock()
        self.test_issue.fields.assignee.displayName = "Test User"
        self.test_issue.fields.reporter = MagicMock()
        self.test_issue.fields.reporter.displayName = "Reporter"
        self.test_issue.fields.created = "2024-01-01T00:00:00.000+0000"
        self.test_issue.fields.updated = "2024-01-02T00:00:00.000+0000"
        self.test_issue.fields.labels = ["test", "bug"]
        self.test_issue.fields.components = ["Component1"]
        self.test_issue.fields.fixVersions = ["1.0.0"]
        self.test_issue.fields.comment = MagicMock()
        self.test_issue.fields.comment.comments = []
        self.test_issue.changelog = MagicMock()
        self.test_issue.changelog.histories = []

    def tearDown(self):
        """Clean up test environment"""
        logger.info("Cleaning up test fixtures")
        # Clean up environment variables
        os.environ.pop("JIRA_URL", None)
        os.environ.pop("JIRA_USERNAME", None)
        os.environ.pop("JIRA_PASSWORD", None)

    @patch('devtoolbox.jira_client.JIRA')
    def test_initialization(self, mock_jira_class):
        """Test JiraClient initialization"""
        logger.info("Testing JiraClient initialization")
        mock_jira_class.return_value = self.mock_jira

        # Test initialization with environment variables
        client = JiraClient()
        self.assertEqual(client.jira_url, os.environ["JIRA_URL"])
        self.assertEqual(client.username, os.environ["JIRA_USERNAME"])
        self.assertEqual(client.password, os.environ["JIRA_PASSWORD"])

        # Test initialization with parameters
        custom_url = "https://custom.atlassian.net"
        custom_username = "custom_user"
        custom_password = "custom_password"
        client = JiraClient(
            jira_url=custom_url,
            username=custom_username,
            password=custom_password
        )
        self.assertEqual(client.jira_url, custom_url)
        self.assertEqual(client.username, custom_username)
        self.assertEqual(client.password, custom_password)

    @patch('devtoolbox.jira_client.JIRA')
    def test_search_issues(self, mock_jira_class):
        """Test search_issues method"""
        logger.info("Testing search_issues method")
        mock_jira_class.return_value = self.mock_jira
        self.mock_jira.search_issues.return_value = [self.test_issue]

        client = JiraClient()
        jql = "project = TEST"
        max_results = 10

        # Test search with max_results
        issues = client.search_issues(jql, max_results)
        self.mock_jira.search_issues.assert_called_once_with(
            jql, maxResults=max_results
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].key, self.test_issue_key)

        # Test search without max_results
        self.mock_jira.search_issues.reset_mock()
        issues = client.search_issues(jql)
        self.mock_jira.search_issues.assert_called_once_with(
            jql, maxResults=False
        )

    @patch('devtoolbox.jira_client.JIRA')
    def test_get_issue_details(self, mock_jira_class):
        """Test get_issue_details method"""
        logger.info("Testing get_issue_details method")
        mock_jira_class.return_value = self.mock_jira
        self.mock_jira.issue.return_value = self.test_issue

        client = JiraClient()

        # Test JSON output format
        details = client.get_issue_details(self.test_issue_key, "json")
        self.assertIsInstance(details, dict)
        self.assertEqual(details["key"], self.test_issue_key)
        self.assertEqual(details["summary"], "Test Issue")
        self.assertEqual(details["status"], "To Do")

        # Test markdown output format
        details = client.get_issue_details(self.test_issue_key, "markdown")
        self.assertIsInstance(details, str)
        self.assertIn(self.test_issue_key, details)
        self.assertIn("Test Issue", details)
        self.assertIn("To Do", details)

    @patch('devtoolbox.jira_client.JIRA')
    def test_update_issue_labels(self, mock_jira_class):
        """Test update_issue_labels method"""
        logger.info("Testing update_issue_labels method")
        mock_jira_class.return_value = self.mock_jira
        self.mock_jira.issue.return_value = self.test_issue

        client = JiraClient()
        new_labels = ["new_label1", "new_label2"]

        # Test updating labels
        client.update_issue_labels(self.test_issue_key, new_labels)
        self.mock_jira.issue.assert_called_once_with(self.test_issue_key)

        # Get the actual arguments passed to update
        call_args = self.test_issue.update.call_args[1]
        updated_labels = call_args['fields']['labels']

        # Verify that all new labels are present
        for label in new_labels:
            self.assertIn(label, updated_labels)

        # Verify that existing labels are preserved
        for label in self.test_issue.fields.labels:
            self.assertIn(label, updated_labels)

    @patch('devtoolbox.jira_client.JIRA')
    def test_get_active_sprints(self, mock_jira_class):
        """Test get_active_sprints method"""
        logger.info("Testing get_active_sprints method")
        mock_jira_class.return_value = self.mock_jira

        # Create mock board and sprint
        mock_board = MagicMock()
        mock_board.id = 123
        mock_board.type = "scrum"
        self.mock_jira.boards.return_value = [mock_board]

        mock_sprint = MagicMock()
        mock_sprint.state = "active"
        self.mock_jira.sprints.return_value = [mock_sprint]

        client = JiraClient()
        sprints = client.get_active_sprints(self.test_project_key)

        self.mock_jira.boards.assert_called_once_with(
            projectKeyOrID=self.test_project_key
        )
        self.mock_jira.sprints.assert_called_once_with(mock_board.id)
        self.assertEqual(len(sprints), 1)
        self.assertEqual(sprints[0], mock_sprint)

    @patch('devtoolbox.jira_client.JIRA')
    def test_create_issue(self, mock_jira_class):
        """Test create_issue method"""
        logger.info("Testing create_issue method")
        mock_jira_class.return_value = self.mock_jira
        self.mock_jira.create_issue.return_value = self.test_issue

        client = JiraClient()
        summary = "Test Issue"
        issue_type = "Story"
        description = "Test Description"

        # Test creating issue
        issue_key = client.create_issue(
            self.test_project_key,
            summary,
            issue_type=issue_type,
            description=description
        )

        self.mock_jira.create_issue.assert_called_once()
        self.assertEqual(issue_key, self.test_issue_key)

    @patch('devtoolbox.jira_client.JIRA')
    def test_update_issue(self, mock_jira_class):
        """Test update_issue method"""
        logger.info("Testing update_issue method")
        mock_jira_class.return_value = self.mock_jira
        self.mock_jira.issue.return_value = self.test_issue

        client = JiraClient()
        new_summary = "Updated Summary"
        new_description = "Updated Description"

        # Test updating issue
        client.update_issue(
            self.test_issue_key,
            summary=new_summary,
            description=new_description
        )

        self.mock_jira.issue.assert_called_once_with(self.test_issue_key)
        self.test_issue.update.assert_called_once()

    @patch('devtoolbox.jira_client.JIRA')
    def test_delete_issue(self, mock_jira_class):
        """Test delete_issue method"""
        logger.info("Testing delete_issue method")
        mock_jira_class.return_value = self.mock_jira
        self.mock_jira.issue.return_value = self.test_issue

        client = JiraClient()

        # Test deleting issue
        result = client.delete_issue(self.test_issue_key)
        self.assertTrue(result)
        self.test_issue.delete.assert_called_once()


if __name__ == '__main__':
    unittest.main()