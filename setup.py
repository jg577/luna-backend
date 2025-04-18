from setuptools import setup, find_packages

setup(
    name="luna",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.2",
        "python-dotenv>=1.0.0",
        "langchain>=0.1.0",
        "langgraph",
        "openai>=1.0.0",
        "langchain-openai>=0.0.5",
        "langchain-core>=0.1.0",
        "langchain-community",
    ],
    author="Luna Team",
    author_email="luna@example.com",
    description="Luna brewery/restaurant analytics package for SQL generation and visualization",
    python_requires=">=3.8",
)
