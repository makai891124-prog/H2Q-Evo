import logging
import subprocess

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """Analyzes code changes in the H2Q geometry kernel repository."""

    def __init__(self):
        pass

    def get_recent_changes(self, num_commits=5):
        """Retrieves a list of recent code changes using Git.

        Args:
            num_commits (int): The number of recent commits to retrieve.

        Returns:
            list: A list of commit messages, or an empty list if no changes are found or Git is not available.
        """
        try:
            # Execute git log command to get recent commit messages
            result = subprocess.run(
                ["git", "log", "--pretty=format:%s", "-n", str(num_commits)],
                capture_output=True,
                text=True,
                check=True
            )
            commit_messages = result.stdout.strip().splitlines()
            return commit_messages

        except FileNotFoundError:
            logger.error("Git not found. Please ensure Git is installed and in your PATH.")
            return []
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running git log: {e}")
            return []
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
            return []


# Example usage (for demonstration and testing):
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    analyzer = CodeAnalyzer()
    recent_changes = analyzer.get_recent_changes()

    if recent_changes:
        print("Recent Code Changes:")
        for change in recent_changes:
            print(f"- {change}")
    else:
        print("No recent code changes found.")