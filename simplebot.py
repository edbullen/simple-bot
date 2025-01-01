import argparse


from bot.RagChain import RagChain

bot = RagChain()

DATA_PATH = "./data/arxiv"


if __name__ == '__main__':
    """
    Option to run in 3x modes - 
        1. load the database with chunks and embeddings from PDFs
        2. query the vector database
        3. query the chat-bot

    """
    parser = argparse.ArgumentParser(description="Simple RAG chatbot: load Embeddings, query Vector DB, query Chat-Bot",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--mode', dest='mode', action='store', help='Mode: load / query / chat',
                        choices=['load', 'query', 'chat'],
                        required=True)
    parser.add_argument('--query', dest='q_string', action='store', help='Query string for Vector DB or Chat-Bot',
                        required=False)
    parser.add_argument('--n', dest='k_int', action='store', help='Number of query results, default: 1',
                        default="1",
                        required=False)
    parser.add_argument('--norag', dest='norag', action='store_true', help='switch off RAG and Vector DB (LLM only)',
                        default=False,
                        required=False)

    args = vars(parser.parse_args())

    if args['mode'] != 'load':
        if not args['q_string']:
            print("A query string or question must be provided for query / chat options")
            exit(2)

    if args['mode'] == 'load':
        bot.run_docs_load(DATA_PATH)
    elif args['mode'] == 'query':
        bot.run_query_db(query_string=args['q_string'], k_int=int(args['k_int']))
    elif args['mode'] == 'chat' and not args['norag']:
        bot.run_rag_chat(query_string=args['q_string'], template="simple_rag_qanda.txt")
    elif args['mode'] == 'chat' and args['norag']:
        bot.run_llm_chat(query_string=args['q_string'])
    else:
        print("Unknown Option")

