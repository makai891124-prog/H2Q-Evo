import os
import logging
from github import Github
from github.GithubException import UnknownObjectException

logger = logging.getLogger(__name__)

class GithubIntegration:
    def __init__(self, repo_name, github_token=None):
        self.repo_name = repo_name
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.g = Github(self.github_token)
        self.repo = self.g.get_repo(self.repo_name)

    def get_file_content(self, file_path, max_lines=1000): # Added max_lines to avoid excessive token usage
        try:
            file_content = self.repo.get_contents(file_path)
            decoded_content = file_content.decoded_content.decode()
            lines = decoded_content.splitlines()
            if len(lines) > max_lines:
                return "".join(lines[:max_lines]) + "\n... (truncated due to length)"
            return decoded_content
        except UnknownObjectException:
            logger.warning(f"File not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error getting file content for {file_path}: {e}")
            return None

    def get_project_context(self, max_files=10, max_lines_per_file=500): #Added limits to reduce token usage.
        context = {}
        contents = self.repo.get_contents("")
        files_read = 0
        for content in contents:
            if content.type == "dir":
                continue
            if not content.name.endswith(".py"):
                continue
            if files_read >= max_files:
                logger.info(f"Reached maximum number of files ({max_files}). Skipping remaining files.")
                break

            file_path = content.path
            file_content = self.get_file_content(file_path, max_lines=max_lines_per_file)  # Use the get_file_content method with line limit

            if file_content:
                context[file_path] = file_content
                files_read += 1

        return context
