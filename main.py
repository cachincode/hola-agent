from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import FunctionTool, QueryEngineTool

# Creting basic tools
# Docstrings are very important for the LLMs to know which one to use
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

# Setup of models
Settings.llm = Ollama(model="llama3.1", request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Create RAG engine and a tool from it
documents = SimpleDirectoryReader("./data").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True, similarity_top_k=10)
budget_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="canadian_budget_2023",
    description="A RAG engine with some basic facts about the 2023 Canadian federal budget."
)

# Initialize the agent
agent = ReActAgent.from_tools(
    tools=[multiply_tool, add_tool, budget_tool],
    verbose=True
)

question = ""
response = agent.chat(question)
print(response)
