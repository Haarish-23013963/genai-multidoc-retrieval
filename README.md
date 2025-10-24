## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Design and build a LlamaIndex agent for retrieving information from multiple research papers. The agent should extract and synthesize relevant details to answer user queries. Its performance will be evaluated based on the conciseness, relevance, and accuracy of its responses to diverse questions. The goal is to create an effective tool for knowledge synthesis across a collection of documents.
### DESIGN STEPS:

### STEP 1:
Install Core Libraries: Install llama-index, llama-hub, transformers, sentence-transformers, faiss-cpu, and numpy to ensure all necessary dependencies for LlamaIndex, text embedding, and FAISS are available.

### STEP 2:
Upload Initial Documents: Use google.colab.files.upload() to prompt the user to select and upload source text files (e.g., docs.txt, docs1.txt) into the environment.

### STEP 3:
Import LlamaIndex Document Type: Import the Document class from llama_index.core to enable the creation of LlamaIndex-compatible document objects.

### STEP 4:
Load Documents into LlamaIndex Format: Iterate through the initially uploaded files, read their content, and encapsulate each file's text into a Document object, storing these objects in a list named documents.

### STEP 5:
Import Embedding and Indexing Libraries: Import SentenceTransformer for generating text embeddings, faiss for vector indexing, numpy for numerical array operations, and os (though os is imported, its usage isn't shown in the provided snippet for the core algorithm).

### STEP 6:
Re-upload/Verify Document Files: Potentially re-upload or re-confirm the presence of the source text files, ensuring they are accessible for subsequent processing steps.

### STEP 7:
Prepare Documents for Embedding: Create a new list of documents by reading the content of the uploaded files directly as raw strings, preparing them for input into the embedding model.

### STEP 8:
Generate Embeddings: Initialize a SentenceTransformer model (specifically "all-MiniLM-L6-v2") and use it to transform the prepared text documents into numerical vector embeddings, converting the output to a NumPy array.

### STEP 9:
Initialize and Populate FAISS Index: Determine the dimensionality of the generated embeddings, then initialize a faiss.IndexFlatL2 (a basic FAISS index for L2 distance similarity) with this dimension, and finally add the computed document embeddings to this FAISS index, making them searchable.

### PROGRAM:
```python

from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

urls = [
    "https://openreview.net/forum?id=odjMSBSWRt",
    "https://openreview.net/forum?id=TVQLu34bdw",
    "https://openreview.net/forum?id=syThiTmWWm",
]

papers = [
    "14257_DarkBench_Benchmarking_D.pdf",
    "9667_Proteina_Scaling_Flow_bas.pdf",
    "6258_Cheating_Automatic_LLM_Be.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

len(initial_tools)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Tell me about Dark Patterns in Large Language Models, "
    "and then tell me about the limitations"
)
response = agent.query("Give me a summary of both PROTEINA and CHEATING AUTOMATIC LLM")
print(str(response))

```

### OUTPUT:
<img width="1206" height="238" alt="image" src="https://github.com/user-attachments/assets/33cbc2ac-9350-4015-9a81-976906f66c0a" />
<img width="1110" height="794" alt="image" src="https://github.com/user-attachments/assets/91493a09-ebe7-4db3-8f25-cd35611f154e" />



### RESULT:
Therefore the code is excuted successfully.
