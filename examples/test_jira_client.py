import os
import logging
from devtoolbox.jira_client import JiraClient
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_issue_lifecycle(jira_client):
    """Test the complete lifecycle of an issue: create, update, and get details."""
    project_key = "PRD"
    try:
        # 1. Create a test issue
        logging.info("1. Creating test issue...")
        issue_key = jira_client.create_issue(
            project_key=project_key,
            summary=f"Test Issue - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="This is a test issue created by automated test",
            issue_type="Bug",  # Using Task type as it's commonly available
            priority="Medium",
            labels=["test", "automated"],
            # Don't set assignee to avoid permissions issues
        )
        logging.info(f"Created issue: {issue_key}")

        # 2. Get and verify issue details
        logging.info("\n2. Getting issue details...")
        issue_details = jira_client.get_issue_details(issue_key)
        logging.info("Initial issue details:")
        logging.info(f"Summary: {issue_details['summary']}")
        logging.info(f"Status: {issue_details['status']}")
        logging.info(f"Labels: {issue_details['labels']}")

        # 3. Update the issue
        logging.info("\n3. Updating issue...")
        jira_client.update_issue(
            issue_key=issue_key,
            summary=f"Updated Test Issue - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="This description was updated by the test",
            priority="High",
            labels=["test", "automated", "updated"]
        )

        # 4. Get and verify updated details
        logging.info("\n4. Getting updated issue details...")
        updated_details = jira_client.get_issue_details(issue_key)
        logging.info("Updated issue details:")
        logging.info(f"Summary: {updated_details['summary']}")
        logging.info(f"Status: {updated_details['status']}")
        logging.info(f"Labels: {updated_details['labels']}")

        # 5. Update status (if possible)
        logging.info("\n5. Attempting to update status...")
        jira_client.update_issue(
            issue_key=issue_key,
            status="In Progress"  # Assuming this status exists
        )

        # 6. Get markdown format
        logging.info("\n6. Getting issue details in markdown format...")
        markdown_details = jira_client.get_issue_details(
            issue_key, 
            output_format="markdown"
        )
        logging.info("Markdown output preview (first 500 chars):")
        logging.info(markdown_details[:500] + "...")

        return issue_key

    except Exception as e:
        logging.error(f"Error in test_issue_lifecycle: {str(e)}")
        raise

def test_sprint_operations(jira_client, issue_key):
    """Test sprint-related operations."""
    try:
        logging.info("\nTesting sprint operations...")
        
        # 1. Try to move to active sprint
        logging.info("1. Moving issue to active sprint...")
        jira_client.update_issue(
            issue_key=issue_key,
            sprint="active"
        )

        # 2. Get sprint information
        logging.info("2. Getting active sprints...")
        active_sprints = jira_client.get_active_sprints("PRD")
        if active_sprints:
            logging.info(f"Active sprints: {[s.name for s in active_sprints]}")
        else:
            logging.info("No active sprints found")

    except Exception as e:
        logging.error(f"Error in test_sprint_operations: {str(e)}")
        raise

def test_issue_cleanup(jira_client, issue_key):
    """Test issue deletion and cleanup."""
    try:
        logging.info("\nTesting issue cleanup...")
        
        # 1. Verify issue exists before deletion
        logging.info(f"1. Verifying issue {issue_key} exists...")
        issue_exists = jira_client.get_issue_details(issue_key) is not None
        if issue_exists:
            logging.info(f"Issue {issue_key} found, proceeding with deletion")
        
            # 2. Delete the issue
            logging.info(f"2. Deleting issue {issue_key}...")
            success = jira_client.delete_issue(issue_key, verify=True)
            
            if success:
                logging.info(f"Successfully deleted issue {issue_key}")
            else:
                logging.warning(f"Issue {issue_key} could not be deleted")
                
            # 3. Verify deletion
            logging.info(f"3. Verifying issue {issue_key} was deleted...")
            try:
                jira_client.get_issue_details(issue_key)
                logging.error(f"Issue {issue_key} still exists!")
                return False
            except Exception:
                logging.info(f"Confirmed issue {issue_key} no longer exists")
                return True
        else:
            logging.warning(f"Issue {issue_key} not found")
            return False

    except Exception as e:
        logging.error(f"Error in test_issue_cleanup: {str(e)}")
        raise

def main():
    """Run all test cases."""
    logging.info("Starting JIRA Client Tests")

    # Initialize JiraClient
    jira_client = JiraClient(
        jira_url=os.getenv("JIRA_URL"),
        username=os.getenv("JIRA_USERNAME"),
        password=os.getenv("JIRA_PASSWORD")
    )

    issue_key = None
    try:
        # Run issue lifecycle test
        issue_key = test_issue_lifecycle(jira_client)
        
        # Run sprint operations test
        test_sprint_operations(jira_client, issue_key)

        logging.info("\nAll feature tests completed successfully!")
        
    except Exception as e:
        logging.error(f"Test suite failed: {str(e)}")
        raise
    
    finally:
        # Cleanup: Always try to delete the test issue
        if issue_key:
            logging.info("\nStarting cleanup...")
            try:
                cleanup_success = test_issue_cleanup(jira_client, issue_key)
                if cleanup_success:
                    logging.info("Cleanup completed successfully")
                else:
                    logging.warning("Cleanup may not have been complete")
            except Exception as e:
                logging.error(f"Error during cleanup: {str(e)}")
        else:
            logging.info("No issue to clean up")

if __name__ == "__main__":
    main()
