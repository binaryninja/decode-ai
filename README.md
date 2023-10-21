# decode-ai
![alt text](https://github.com/decode-ai/blob/main/utils/static/ProjectLupine.png?raw=true)

# Plugins:


## Utilities

# api_server.py
## Overview

This script provides an API for malware analysis. It uses FastAPI, SQLAlchemy for database operations, the Transformers library to load our local model and generate output, and langchain for interacting with OpenAI models. The core functionality is to analyze a given piece of decompiled code and provide descriptive information, such as a function summary and a new function name.

## Dependencies

- **fastapi**: Web framework to build the API.
- **pydantic**: Data validation and settings management using Python type annotations.
- **sqlalchemy**: Database toolkit for Python.
- **databases**: Asynchronous database support for FastAPI.
- **torch**: PyTorch library, primarily for deep learning.
- **datasets**: To load datasets.
- **transformers**: For using loading local model and generating content
- **openai**: OpenAI Python client for generating content
- **langchain**: A module possibly related to language processing.
- **json, re, os, time**: Standard libraries for various functionalities.

## Configuration

- **DATABASE_URL**: The SQLite database URL.
- **openai_key**: The API key for OpenAI.
- **model & tokenizer**: Pre-trained models loaded using Transformers library.

## Database Schema

- **Function**: Represents the function table in the database with fields for ID, input, output, review status, and details related to the LLM analysis.
- **EvaluationLog**: Logs request and response data for evaluations.
- **Suggestion**: Contains suggestions for code analysis.

## API Endpoints

1. **`/` (GET)**: Returns the main index page.
2. **`/sample` (GET)**: Provides a sample from the dataset.
3. **`/evaluate` (POST)**: Evaluates the given code using the LLM model and returns analysis results.
4. **`/evaluate_local` (POST)**: Evaluates the given code locally and returns analysis results.
5. **`/approved` (GET)**: Returns a list of approved items.
6. **`/review` (GET)**: Displays items for review based on category.
7. **`/update-output` (POST)**: Updates the output for a given function ID.
8. **`/suggest` (POST)**: Stores a suggestion in the Suggestion table.
9. **`/action` (POST)**: Performs actions like approve or remove on a function.

## Helper Functions

- **`flatten_list`**: Flattens nested lists or dictionaries.
- **`get_llm_name_with_retry`**: Attempts to get the LLM name with retries.
- **`generate_decode_response`**: Generates response using the decode-llm model.
- **`extract_function_info`**: Extracts function summary and new function name from the provided content.
- **`llm_rename_function`**: Renames the function using LLM.
- **`load_data_to_db`**: Loads data to the database.

## Execution

To run the FastAPI application:

```bash
uvicorn.run(app, host="0.0.0.0", port=8000)
```

This will start the server on `http://0.0.0.0:8000`.

---

### populate_db.py
This script loads the decompiled code from the sqlite3 database generated by the web application.

It calls openai to generate llm names, short sum and step by step descriptions.
