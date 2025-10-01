from langchain_ollama import OllamaLLM

# Initialize the LLM with specific options
llm = OllamaLLM(
    model="llama3.1:8b",
)

# Generate text from a prompt
text = """
Write a quick sort algorithm in Python with detailed comments:
```python
def quicksort(
"""

response = llm.invoke(text)
print(response[:500])

