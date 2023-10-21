# Project Lupine
![alt text](https://github.com/binaryninja/decode-ai/blob/main/utils/static/ProjectLupine.png?raw=true)

The Lupine project focus on training large language models to understand decompiled code and reverse engineer binaries.  The models takes decompiled C as input and outputs a descriptive function name, functiona summary, and a step-by-step read out of what the code is doing.

Currently, the project's best model is a finetune version of the Code Llama 34B Instruct model.  It is fine tuned on a dataset of 60,000 decompiled functions extracted from 5000 malware samples and 100 unique malware families.

The model is used by out Ghidra plugin to provide function names, and descriptions directly in your IDE.  A feedback mechanism lets you contribute your knowledge and skills back to the collective for assimilation into the model (yonik!)

Below is a rough architecture
![alt text](https://github.com/binaryninja/decode-ai/blob/main/utils/static/flowchart.png?raw=true)


# Plugins:
There are three Ghidra plugins each with their own configured shortcut keys.

## llm.py (CTRL-ALT-L)
This script calls your local LLM.  It requests a new function name and function description.  It renames the function and updates the somments with the description.  Your cursor can be anywhere inside the decompiled function that you're interested in.

The plugin expects api_server.py to be running on localhost on port 8000.  Documentation for the API server can be found below.

## llm_remote.pt (CTRL-ALT-O)
This script calls the Project Lupine community server.  It requests a new function name and function description.  It renames the function and updates the somments with the description.  Note that it sends the hash, function offset, and decompiled code to the community server.  

## llm_suggest (CTRL-SHIFT-K)
This script is useful for contributing back to the community.  If you get a summary, function name or step-by-step description that you don't like you can edit the content directly in Ghidra and send your edits back.  Note that it sends the hash, function offset, and decompiled code, function name, function comment to the community server.  The 

# Utilities
This section contains the api_server and several helper utilities for loading data and testing the model.

## api_server.py
### Overview

This script provides an API for malware analysis. It uses FastAPI, SQLAlchemy for database operations, the Transformers library to load our local model and generate output, and langchain for interacting with OpenAI models. The core functionality is to analyze a given piece of decompiled code and provide descriptive information, such as a function summary and a new function name.

### Dependencies

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

### Configuration

- **DATABASE_URL**: The SQLite database URL.
- **openai_key**: The API key for OpenAI.
- **model & tokenizer**: Pre-trained models loaded using Transformers library.

### Database Schema

- **Function**: Represents the function table in the database with fields for ID, input, output, review status, and details related to the LLM analysis.
- **EvaluationLog**: Logs request and response data for evaluations.
- **Suggestion**: Contains suggestions for code analysis.

### API Endpoints

1. **`/` (GET)**: Returns the main index page.
2. **`/sample` (GET)**: Provides a sample from the dataset.
3. **`/evaluate` (POST)**: Evaluates the given code using the LLM model and returns analysis results.
4. **`/evaluate_local` (POST)**: Evaluates the given code locally and returns analysis results.
5. **`/approved` (GET)**: Returns a list of approved items.
6. **`/review` (GET)**: Displays items for review based on category.
7. **`/update-output` (POST)**: Updates the output for a given function ID.
8. **`/suggest` (POST)**: Stores a suggestion in the Suggestion table.
9. **`/action` (POST)**: Performs actions like approve or remove on a function.

#### Helper Functions

- **`flatten_list`**: Flattens nested lists or dictionaries.
- **`get_llm_name_with_retry`**: Attempts to get the LLM name with retries.
- **`generate_decode_response`**: Generates response using the decode-llm model.
- **`extract_function_info`**: Extracts function summary and new function name from the provided content.
- **`llm_rename_function`**: Renames the function using LLM.
- **`load_data_to_db`**: Loads data to the database.

### Execution

To run the FastAPI application:

```bash
uvicorn.run(app, host="0.0.0.0", port=8000)
```

This will start the server on `http://0.0.0.0:8000`.

---

### populate_db.py
This script loads the decompiled code from the sqlite3 database generated by the web application.

It calls openai to generate llm names, short sum and step by step descriptions.
