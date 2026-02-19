```markdown
# Intro to Large Language Models (LLMs)

## What is an LLM?
Large Language Models are neural models (usually transformer-based) trained on large-scale text corpora to model language. They predict tokens given context and produce rich contextual representations used for generation, understanding, and downstream tasks.

## Core Architecture: Transformer
- **Self-attention:** Each token attends to others using queries, keys, and values. Attention weights are computed by scaled dot‑product and allow modeling of long‑range dependencies efficiently.

  **Scaled dot‑product attention (pseudo‑code):**
  ```python
  import torch
  def attention(Q, K, V):
      # Q, K, V: [seq_len, d]
      scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
      weights = torch.softmax(scores, dim=-1)  # [seq_len, seq_len]
      return torch.matmul(weights, V), weights
  ```

- **Multi-head attention:** Several attention blocks in parallel; each head has its own Q/K/V projections. Concatenate head outputs and project again.
- **Feed-forward layers:** Position-wise MLPs applied after attention blocks, usually with ReLU or GELU activation.
- **Residual connections & normalization:** Add input to output of sub-layer (`x + sublayer(x)`) and apply layer normalization; improves gradient flow and stabilizes training.

## Training Paradigms
- **Causal / Autoregressive (e.g., GPT):** Model p(x_t | x_{<t}) and used for free-form generation.
- **Masked / Denoising (e.g., BERT):** Mask tokens and predict them; good for encoding and classification.
- **Seq2Seq pretraining (e.g., T5):** Encoder-decoder models trained on denoising/objective tasks allowing flexible generation.

## Tokenization
- Subword tokenizers (Byte-Pair Encoding, WordPiece, SentencePiece) split text into manageable tokens balancing vocabulary size vs sequence length.
- Tokenization affects model behavior; always use the tokenizer associated with a pretrained model.

**Example using Hugging Face**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = "Hello, how are you?"
tokens = tokenizer(text)
print(tokens)
print(tokenizer.convert_ids_to_tokens(tokens['input_ids']))
# ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']
```

Notice the special prefix (`Ġ`) indicating a space, a peculiarity of GPT‑2's byte-level BPE.

## Capabilities & Use Cases
- Text generation, summarization, translation, question answering, code generation, assistants, semantic search (via embeddings), and more.

**Text generation example:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids
outputs = model.generate(input_ids, max_length=50, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

This runs autoregressive generation: each new token conditions on previous ones.

## Fine-tuning vs In-Context Learning
- **Fine-tuning:** Update model weights on a labeled dataset for a specific task; resource-intensive but often yields best task performance.

  ```python
  # simple finetune with HuggingFace Trainer
  from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
  model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
  # prepare dataset with input_ids and labels...
  trainer = Trainer(model=model, args=TrainingArguments(output_dir='./out', num_train_epochs=3), train_dataset=train_ds)
  trainer.train()
  ```

- **In-Context Learning / Prompting:** Provide examples in the prompt to guide behavior without weight updates. Powerful for few-shot tasks but has limits (context window size, brittleness).

  ```python
  prompt = "Translate English to French:\nEnglish: Hello\nFrench: Bonjour\nEnglish: How are you?\nFrench:"
  # attach to generation as above
  ```

  The model sees examples in the prompt and continues the pattern.

## Scaling & Emergent Behavior
- Larger models and larger pretraining datasets often yield qualitatively new capabilities (emergent behaviors). Scaling improves few-shot performance and robustness but increases compute and alignment challenges.

## Practical Engineering Notes
- **Context window:** LLMs have finite context lengths; use retrieval-augmented generation (RAG) to provide external knowledge beyond the window.
- **Safety & Bias:** LLMs reflect training data and can produce biased, toxic, or incorrect outputs. Apply filtering, human review, and alignment techniques.
- **Latency & Cost:** Larger models have higher inference costs; consider distillation, quantization (8-bit/4-bit), or smaller specialized models.
- **Evaluation:** Use human evaluation, task-specific metrics, and red-teaming for safety-critical systems.

## Deployment Patterns
- **Server-side inference:** Centralized model serving (GPU/TPU) for high-quality responses.
- **Local / Edge:** Tiny or quantized models for on-device privacy-sensitive use.
- **Hybrid (RAG):** Retrieve relevant documents, then condition generation on retrieved context for up-to-date or factual outputs.

## Limitations
- **Hallucinations:** Confident but incorrect outputs; mitigations include grounding with retrieval and answer verification.
- **Context length limits:** Hard limits require chunking or retrieval strategies.
- **Data privacy:** Pretrained models may memorize sensitive data if present in training corpora.

## Next Steps for Learning
- Study the transformer paper (Vaswani et al., 2017).
- Experiment with small transformer implementations (PyTorch, Hugging Face Transformers).
- Try prompt engineering, RAG, and fine-tuning small models to gain practical experience.

```
