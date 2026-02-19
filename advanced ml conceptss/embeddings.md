```markdown
# Embeddings — What They Are and How to Use Them

## Definition
An embedding is a dense, low-dimensional vector representation of a discrete object (word, token, image patch, user, item) that captures semantic relationships by geometry: similar objects have vectors close under a chosen metric (commonly cosine similarity or Euclidean distance).

## Why Embeddings?
- Convert categorical, textual, or high-dimensional inputs into continuous vectors usable by ML models.
- Capture semantic relationships (e.g., `vec('king') - vec('man') + vec('woman') ≈ vec('queen')`).

**Example arithmetic:** with pre-trained Word2Vec vectors you can demonstrate the classic analogy:

```python
from gensim.downloader import load
model = load('word2vec-google-news-300')  # pre-trained vectors (~1.6GB)
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)  # [('queen', 0.7118)]
```

This shows the geometry: adding and subtracting vectors corresponds to semantic composition.

## How Embeddings Are Learned
- **Supervised learning:** Train embeddings together with a downstream task (classification, recommendation).
- **Self-supervised / contrastive:** Learn to pull positive pairs together and push negatives apart (e.g., SimCLR, triplet loss).
- **Language-model-based:** Contextual embeddings come from internal layers of language models; static embeddings (Word2Vec, GloVe) are trained with co-occurrence or predictive objectives.

## Embedding Types
- **Static embeddings:** Single vector per token (Word2Vec, GloVe).
- **Contextual embeddings:** Vary by context (BERT, GPT); capture polysemy.
- **Learned task-specific embeddings:** Trained end-to-end for recommendation or classification.

## Distance & Similarity
- **Cosine similarity:** Common for measuring semantic similarity; invariant to vector length.
- **Euclidean distance:** Useful when absolute magnitudes matter.
- **Dot product:** Often used inside models (e.g., attention scores). Normalize if necessary.

## Practical Uses
- **Semantic search / retrieval:** Encode queries and documents, retrieve by nearest neighbors (ANN indexes like FAISS, Annoy, Milvus). For example, compute text embeddings with a pretrained model and then use cosine similarity to find the closest document.

  ```python
  from sentence_transformers import SentenceTransformer
  import numpy as np

  model = SentenceTransformer('all-MiniLM-L6-v2')
  docs = ['cat video', 'machine learning tutorial', 'buy cheap shoes']
  doc_embs = model.encode(docs)

  query = 'how to train a neural net'
  q_emb = model.encode([query])[0]
  sims = np.dot(doc_embs, q_emb) / (np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb))
  print('closest doc:', docs[np.argmax(sims)])
  ```

- **Clustering & visualization:** Reduce embedding dimensionality (PCA/UMAP) for inspection. Visualize with scatter plots to see semantic groupings.
- **Recommendation:** User/item embeddings for collaborative filtering (matrix factorization, deep learning). Example: multiply user and item vectors to predict ratings.
- **Transfer learning:** Use pretrained embeddings as features for downstream tasks.

## Engineering Considerations
- **Dimensionality:** 50–300 for classical embeddings; 512–4096 for modern contextual representations. Balance quality vs storage/latency.
- **Indexing:** For large collections, use approximate nearest neighbor (ANN) search to scale (HNSW, IVF, PQ).
- **Normalization:** Normalize vectors (unit length) before cosine search to speed up comparisons.
- **Updating embeddings:** For dynamic domains, support incremental retraining or hybrid approaches (cold-start heuristics).

## Common Pitfalls
- Using raw token vectors from different models without alignment causes incompatibility.
- Ignoring distributional shifts — embeddings trained on one domain may not transfer well to another.
- Storing massive embedding corpora without compression or ANN can be costly.

## Quick Examples
- **Word2Vec / Skip-gram:** Predict context words from a center word; produces static embeddings. Training example:

  ```python
  from gensim.models import Word2Vec
  sentences = [['the', 'cat', 'sat'], ['the', 'dog', 'barked']]
  w2v = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=4)
  print(w2v.wv['cat'])  # 50-dim vector
  ```

- **Contrastive loss:** Learn `sim(a, b)` high for positives and low for negatives using InfoNCE or triplet losses. Pseudocode:

  ```python
  # similarity = dot(normalize(a), normalize(b))
  loss = -log( exp(sim(a, b_p)) / (exp(sim(a, b_p)) + sum(exp(sim(a, b_n)) for b_n in negatives)) )
  ```

- Sentence transformers / CLIP-style training: encode sentences and images jointly, then train with contrastive loss so matching pairs have high cosine similarity.

```
