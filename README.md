# RAG Application

This is an experiment to host local Code assist with Opensource components based on Retrieval-Augmented Generation (RAG).

## How to Use
To use this project, you need to set the following environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `GITHUB_ACCESS_TOKEN`: Your GitHub access token.


## Prerequisites
Before using this project, make sure you have the following:
- OpenAI API key
- GitHub access token

## How to customize
The project allows you to change the below parameters through the config.yaml file
MODEL_NAME: The name of the RAG model to use for code generation.
TOP_K: The number of top-k candidates to consider during code generation.
TOP_P: The cumulative probability threshold for nucleus sampling during code generation.
MAX_LENGTH: The maximum length of the generated code.
TEMPERATURE: The temperature value for controlling the randomness of the generated code.
