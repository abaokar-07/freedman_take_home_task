import os
from setuptools import setup, find_packages

setup(
packages = find_packages(),
name = 'doc_qa',
install_requires= ['transformers', 'sentence-transformers', 'langchain', 'langchain-community', 'huggingface-hub', 'pypdf', 'gradio'],
)