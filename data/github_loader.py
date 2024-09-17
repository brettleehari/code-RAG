import requests
import os
import yaml
import fire
import base64
from models.code_chunker import LlamaDoc




class GitHubRepoLoader:
    def __init__(self, config_file='../config/config.yaml'):
        self._parse_config(config_file)
        self.access_token = os.getenv('GITHUB_ACCESS_TOKEN')
        self.api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/contents"

    def _get_headers(self):
        headers = {}
        if self.access_token:
            headers['Authorization'] = f'token {self.access_token}'
        return headers

    def _parse_config(self, config_file):
        """Parse the config file and assign instance variables."""
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.repo_owner = config.get('repository', {}).get('owner')
            self.repo_name = config.get('repository', {}).get('name')

    def get_file_list(self, path=""):
        """Fetch the list of files in the given path of the repository."""
        url = f"{self.api_url}/{path}"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def get_file_data(self, file_path):
        """Fetch the content of a specific file in the repository."""
        status = False
        url = f"{self.api_url}/{file_path}"
        file_extension = os.path.splitext(file_path)[1]
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        file_info = response.json()
        lines = base64.b64decode(file_info['content']).decode('utf-8').splitlines()
        num_lines = len(lines)
        file_info['lines_of_code'] = num_lines
        file_info['extension'] = file_extension
        file_info['content'] = lines
        if 'content' in file_info:
            status = True
        return status, file_info

    
    def traverse_repo(self, path=""):
        """Traverse the repository and produce a graph of each file and folder."""
        graph = {self.repo_name: {}}
        llama_doc = []
        items = self.get_file_list(path)
        for item in items:
            if item['type'] == 'dir':
                graph[self.repo_name][item['name']] = self.traverse_repo(item['path'])
            else:
                graph[self.repo_name][item['name']] = 'file'
                status, content = self.get_file_data(item['path'])
                if status:
                    llama_doc.append(content)
        return graph, llama_doc


if __name__ == '__main__':
    graph, modules = GitHubRepoLoader().traverse_repo()
    doc = LlamaDoc(graph, modules)
    doc.create_doc()
    fire.Fire(GitHubRepoLoader)
