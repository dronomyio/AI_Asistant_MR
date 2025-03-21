from setuptools import setup, find_packages

setup(
    name="modalai_docs_retrieval",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "anthropic",
        "voyageai",
        "requests",
        "beautifulsoup4",
        "tqdm",
        "numpy",
        "pandas",
        "cohere",
        "elasticsearch>=8.0.0",
        "weaviate-client>=3.15.0",
        "docker"
    ],
)