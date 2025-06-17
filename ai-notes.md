# AI Learning Notes (Embeddings, LLMs, RAG)

## ✅ 1. Embeddings + Similarity

- Text embeddings convert sentences into high-dimensional vectors.
- Cosine similarity can measure semantic similarity between two text inputs.

Example:
```text
“How are you?” vs “What’s up?” → high similarity
“How are you?” vs “Banana”     → low similarity
```

## ✅ 2. Retrieval-Augmented Generation (RAG)

**RAG** is a design pattern that enhances LLM responses using relevant context from a knowledge base or dataset.  
Instead of relying only on the model's training, it retrieves relevant information and injects it into the prompt dynamically.

---

### 🔄 RAG Workflow

```text
User Query
   ↓
Embed the Query (OpenAI / HuggingFace model)
   ↓
Compare with Vector Store (cosine similarity)
   ↓
Retrieve Top-K Relevant Chunks
   ↓
Construct Prompt:
  - Inject Retrieved Context
  - Append User Question
   ↓
Send to LLM (e.g., GPT-3.5 / 4)
   ↓
LLM Generates Answer Using Both
