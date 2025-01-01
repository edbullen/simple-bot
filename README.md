# Simple LLM Chatbot

RAG Chatbot based on [Langchain](https://python.langchain.com/docs/introduction/).  
Based on Python 3.10  
First release uses the OpenAI API `ChatOpenAI()` model and [Chroma](https://docs.trychroma.com/docs/overview/introduction) DB as a vector database.

### File Structure 

```
--
 |
  -- bot/RagChain.py
 |
  -- chroma/<database files>
 |
  -- data/<raw PDF data-source>
 |
  -- logs/<chatbot logs>
 |
  -- templates/<chat template files>
```

### Running a Bot Instance

Set the OpenAI API Key:
```
export OPENAI_API_KEY=<sk-proj-key>
```
Instantiate a Bot
```
from bot.RagChain import RagChain
bot = RagChain()
```
Refer to the bot instance methods for using the bot, or instead just use the CLI.  Subset of methods:
```
load_template()
query_rag()
query_chroma()
```

# Utils and Helpers

### `simplebot.py` CLI  

CLI to populate the vector database, query the vector database and query the RAG chatbot. Usage:

```
usage: simplebot.py [-h] --mode {load,query,chat} [--query Q_STRING] [--n K_INT]

Simple RAG chatbot: load Embeddings, query Vector DB, query Chat-Bot

options:
  -h, --help            show this help message and exit
  --mode {load,query,chat}
                        Mode: load / query / chat
  --query Q_STRING      Query string for Vector DB or Chat-Bot
  --n K_INT             Number of query results, default: 1

```

### Example - run the RAG Chatbot CLI

```
`python simplebot.py --mode chat --query "What is the difference between an LTSM model and a Transformer model?"
```
```
The main difference between an LSTM model and a Transformer model is that the Transformer model introduced the
 self-attention mechanism, which allows the model to process all words in a sequence simultaneously, in parallel, 
 rather than sequentially. This parallelization dramatically increased training efficiency and speed, enabling 
 the model to scale effectively to large datasets. Additionally, the self-attention mechanism allowed the Transformer
 to capture long-range dependencies in text more effectively than LSTMs.
``` 

### Example - run the Chatbot CLI in "No-RAG" Mode

This bypasses the Chroma DB and just gets the response directly from the LLM.
```
python simplebot.py --mode chat --norag --query "What is the difference between an LTSM model and a Transformer model?"
```

```
Long Short-Term Memory (LSTM) and Transformer are both popular models used in natural language processing tasks, but they have some key differences:

1. Architecture:
- LSTM is a type of recurrent neural network (RNN) that is designed to capture long-term dependencies in sequential data. It consists of recurrent units with a gating mechanism that allows it to remember information over long periods of time.
- Transformer, on the other hand, is a feedforward neural network architecture that does not rely on recurrence or convolution. It uses self-attention mechanisms to capture dependencies between different parts of the input sequence.

2. Performance:
- Transformers have been shown to outperform LSTMs on many natural language processing tasks, especially on tasks that require capturing long-range dependencies. This is because Transformers are able to process input sequences in parallel, while LSTMs process input sequentially.
- LSTMs are still widely used and have shown good performance on tasks where sequential modeling is important, such as speech recognition and language modeling.

3. Training:
- Transformers are easier to train compared to LSTMs, as they do not suffer from vanishing or exploding gradient problems that can often occur in RNNs. Transformers also require less training time due to their parallel processing capabilities.
- LSTMs can be more challenging to train, especially on tasks that require capturing long-term dependencies, as they can struggle with vanishing gradients and forget important information over time.

In summary, the main differences between LSTM and Transformer models lie in their architecture, performance, and training characteristics. Transformers are generally preferred for tasks that require capturing long-range dependencies and can be easier to train, while LSTMs are still widely used for tasks where sequential modeling is important.
```

# Setup

1. Set the `OPENAI_API_KEY` as an exported shell variable (or associate it with the process running `RagChain.py`).  See *OpenAPI API Access* below for getting an OpenAPI account.
2. Create the `data` folder and populate it with source data in PDF format.  This is for the RAG expertise to add to the base LLM chabot
3. Create the `chroma` folder and populate it with embeddings generated from the chunked-up sections of PDFs in the data folder

### OpenAI API Access

Get an account set up at [platform.openai.com](https://platform.openai.com)
+ You can create API keys at a user or service account level.
+ Keys are scoped to a *project* (preferred) or *user* (legacy)
+ API requests need to include the API key in an Authorization HTTP header: `Authorization: Bearer OPENAI_API_KEY`
+ Further details at [platform.openai.com/docs/api-reference/authentication](https://platform.openai.com/docs/api-reference/authentication)

Generate a platform key by going to *settings* (top right in web GUI) -> *API keys* -> *Create New Secret Key*  

### Download PDFs for processing to `data`

Choose a set of PDFs relevant to the specialist topic for the RAG chatbot.  
  
Example: script to download research papers about Machine Learning and AI:  
`download_arxiv.py` - download the [arXiv](https://arxiv.org/) papers to a local folder, based on a search-criteria filter.

### Populate the Chroma DB from PDFs

The `simplebot.py` CLI can be used to populate the Chroma database in `chroma` from the raw PDF data in `/data`.  
Run Chroma load:
```
python simplebot.py --mode load
```
Log output:
```
2024-12-27 10:24 INFO: Initialized logging
2024-12-27 10:24 INFO: Loading PDF documents
2024-12-27 10:26 INFO: Loaded 1805 PDF pages
2024-12-27 10:26 INFO: Splitting text into chunks
2024-12-27 10:26 INFO: Generating embeddings and saving text chunks and embeddings to vector database
2024-12-27 10:26 INFO: ChromaDB database path: ./chroma
2024-12-27 10:26 INFO: Deleting previous ChromaDB database
2024-12-27 10:28 INFO: Saved 10049 embeddings

```