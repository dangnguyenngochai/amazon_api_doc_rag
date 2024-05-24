from typing import AsyncIterator, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import json
import yaml

class JsonLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        with open(self.file_path, encoding="utf-8") as f:
            json_data = json.load(f)
            # extracting paths
            for item in json_data['paths'].items():
                item_str = str(item)
                yield Document(
                    page_content=item_str,
                    metadata={"source": self.file_path},
                )
            
            # extracting components
            for k,v in json_data['components'].items():
                if k == 'schemas' or k == 'reponses' or k == 'parameters':
                    for item in v.items():
                        item_str = str(item)
                        yield Document(
                                    page_content=schema_item_str,
                                    metadata={"source": self.file_path},
                        )

class YamlLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        with open(self.file_path, encoding="utf-8") as f:
            yaml_data = yaml.load(f)
            # extracting paths
            for item in yaml_data['paths'].items():
                item_str = str(item)
                yield Document(
                    page_content=item_str,
                    metadata={"source": self.file_path},
                )
            
            # extracting components
            for k,v in yaml_data['components'].items():
                if k == 'schemas' or k == 'reponses' or k == 'parameters':
                    for item in v.items():
                        item_str = str(item)
                        yield Document(
                                    page_content=item_str,
                                    metadata={"source": self.file_path},
                        )
          