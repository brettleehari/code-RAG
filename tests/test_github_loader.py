import unittest
from unittest.mock import patch, MagicMock
from data.github_loader import GitHubRepoLoader
class TestGitHubRepoLoader(unittest.TestCase):

    def setUp(self):
        self.repo_owner = "test_owner"
        self.repo_name = "test_repo"
        self.access_token = "test_token"
        self.loader = GitHubRepoLoader(self.repo_owner, self.repo_name, self.access_token)

    @patch('src.github_loader.requests.get')
    def test_get_file_list(self, mock_get):
        # Mock the response from the GitHub API
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"name": "file1.txt", "path": "file1.txt"},
            {"name": "file2.txt", "path": "file2.txt"}
        ]
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Call the method
        file_list = self.loader.get_file_list()

        # Assertions
        self.assertEqual(len(file_list), 2)
        self.assertEqual(file_list[0]['name'], 'file1.txt')
        self.assertEqual(file_list[1]['name'], 'file2.txt')

    def test_get_headers_with_token(self):
        headers = self.loader._get_headers()
        self.assertIn('Authorization', headers)
        self.assertEqual(headers['Authorization'], f'token {self.access_token}')

    def test_get_headers_without_token(self):
        loader = GitHubRepoLoader(self.repo_owner, self.repo_name)
        headers = loader._get_headers()
        self.assertNotIn('Authorization', headers)

if __name__ == '__main__':
    unittest.main()