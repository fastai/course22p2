import sys
from langchain import OpenAI
from pathlib import Path
import llama_index as li
#from llamahub.connectors import TextFileConnector
from llama_index import SimpleDirectoryReader,GPTListIndex,LLMPredictor

file_name = sys.argv[1]
llm_predictor = LLMPredictor(llm=OpenAI(model_name="gpt-3.5-turbo")) #temperature=0, 
docs = SimpleDirectoryReader('.', [file_name]).load_data()
index = GPTListIndex(docs)
ex = """Today we finish off our study of collaborative filtering by looking closely at embeddings—a critical building block of many deep learning algorithms. Then we’ll dive into convolutional neural networks (CNNs) and see how they really work. We’ve used plenty of CNNs through this course, but we haven’t peeked inside them to see what’s really going on in there. As well as learning about their most fundamental building block, the convolution, we’ll also look at pooling, dropout, and more."""
q = f"""Here's an example of a lesson summary from a previous fast.ai lesson: "{ex}" Write a four paragraph summary of the fast.ai lesson contained in the following transcript, using a similar informal writing style to the above summary from the previous lesson."""

summary = index.query(q, response_mode="tree_summarize", llm_predictor=llm_predictor)
Path(f'{Path(file_name).stem}-summ.txt').write_text(str(summary))

