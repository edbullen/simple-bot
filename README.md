# Simple LLM Chatbots

## Utils and Helpers

`download_arxiv.py` - download the [arXiv](https://arxiv.org/) papers to a local folder, based on a search-criteria filter.

### OpenAI API Access

Get an account set up at [platform.openai.com](https://platform.openai.com)
+ You can create API keys at a user or service account level.
+ Keys are scoped to a *project* (preferred) or *user* (legacy)
+ API requests need to include the API key in an Authorization HTTP header: `Authorization: Bearer OPENAI_API_KEY`
+ Further details at [platform.openai.com/docs/api-reference/authentication](https://platform.openai.com/docs/api-reference/authentication)

Generate a platform key by going to *settings* (top right in web GUI) -> *API keys* -> *Create New Secret Key*  

## Simple-Bot 1: OpenAI + Chroma RAG ChatBot

`openai_chroma_rag` - 