import os
import yaml

from llama_index.core.ingestion import IngestionPipeline

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter


from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor, TitleExtractor

from llama_index.core.metadata_mode import MetadataMode


class MetadataExtractors:
    def __init__(self, document, config_file='../config/config.yaml'):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self._parse_config(config_file)
        self.document = document


    def _parse_config(self, config_file):
        """Parse the config file and assign instance variables."""
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.repo_owner = config.get('repository', {}).get('owner')
            self.repo_name = config.get('repository', {}).get('name')

    def extract_metadata(self):

        text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
        title_extractor = TitleExtractor(nodes=5)
        qa_extractor = QuestionsAnsweredExtractor(questions=3)
        pipeline = IngestionPipeline(transformations=[text_splitter, title_extractor, qa_extractor])
        nodes = pipeline.run(documents=self.document, in_place=True, show_progress=True)
        return nodes


"""
        self.node_parser = TokenTextSplitter(
            separator=" ", chunk_size=256, chunk_overlap=128
        )
        self.extractors_1 = [
            QuestionsAnsweredExtractor(
                questions=3, llm=self.llm, metadata_mode=MetadataMode.EMBED
            ),
        ]
        self.extractors_2 = [
            SummaryExtractor(summaries=["prev", "self", "next"], llm=self.llm),
            QuestionsAnsweredExtractor(
                questions=3, llm=self.llm, metadata_mode=MetadataMode.EMBED
            ),
        ]
"""