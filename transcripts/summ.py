import sys
import openai
import llama_index as li
from llamahub.connectors import TextFileConnector
from llama_index.indices import GPTListIndex

file_name = sys.argv[1]
#openai.api_key = '<your-api-key>'
index = GPTListIndex()
connector = TextFileConnector(file_name)
index.load_data(connector)
summary = index.query(response_mode="tree_summarize", llm="gpt-3.5-turbo", summary_length=100)
print(summary)

