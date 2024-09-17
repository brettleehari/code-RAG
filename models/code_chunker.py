from llama_index.core.schema import Document
from datetime import datetime
import time
import yaml

class LlamaDoc:
    def __init__(self, graph, modules, config_file='../config/config.yaml'):
        self.modules = modules
        self.graph = graph
        self.doc = []
        self._parse_config(config_file)

    def _parse_config(self, config_file):
        """Parse the config file and assign instance variables."""
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.repo_owner = config.get('repository', {}).get('owner')
            self.repo_name = config.get('repository', {}).get('name')

    def create_doc(self):
        for module in self.modules:
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
            document = Document(text=module['content'],
            metadata={
                "file_name": module["name"],
                "githubrepo": self.repo_name+"/"+self.repo_owner,
                "extension": module["extension"],
                "modifiedOn": formatted_now,
                "size": module["size"],
                "github_url": module["html_url"],
                "lines": module["lines_of_code"],},
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}")
            time.sleep(3)
            self.doc.append(document)
        return self.doc
