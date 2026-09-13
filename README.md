# Customer Support Chatbot Task

This repository contains an end-to-end NLP customer-support chatbot prototype in [ahmadtaher_nlp_final_task_rest_of_modules.ipynb](ahmadtaher_nlp_final_task_rest_of_modules.ipynb). The notebook is organized as a four-stage pipeline that combines supervised text classification with retrieval-augmented generation (RAG).

## Architecture

```text
Customer message
		 |
		 v
1. Language detection
	Character n-gram TF-IDF + linear classifier
		 |
		 v
2. Sentiment / emotion classification
	TF-IDF baseline, BiLSTM, and DistilBERT comparison
		 |
		 v
3. Intent classification
	27 source intents mapped to 7 routing buckets
	TF-IDF + Logistic Regression and tuned LinearSVC
		 |
		 v
4. Q&A response generation
	Sentence-Transformer embeddings -> FAISS top-k retrieval
	Retrieved support examples + metadata -> Groq LLM response
```

The first three stages produce structured routing signals. The fourth stage uses those signals, the original customer message, and retrieved examples from the Bitext support dataset to generate a grounded answer. Negative sentiment combined with a complaint intent receives an empathetic routing treatment before generation.

## Components

### 1. Language detection

The notebook loads `papluca/language-identification`, which contains 20 languages. It compares a word-level TF-IDF baseline with character n-gram TF-IDF models and a multilingual DistilBERT classifier. Character features preserve script and short language-specific patterns, making them a strong, inexpensive baseline for short messages. The notebook includes validation, test metrics, a confusion matrix, and focused Arabic/Urdu error analysis.

### 2. Sentiment and emotion

The `dair-ai/emotion` dataset supplies six emotion labels: sadness, anger, fear, joy, love, and surprise. These are also mapped to three routing buckets: `negative`, `neutral`, and `positive`. The module compares TF-IDF plus Logistic Regression, a from-scratch PyTorch BiLSTM, and DistilBERT. A small hand-labelled support-style set provides a qualitative domain-shift check because the source data is based on Twitter text rather than customer-support conversations.

### 3. Intent classification

The Bitext customer-support dataset contains fine-grained intents and categories. The notebook maps its 27 intents into seven coarse routing buckets, creates stratified train/validation/test splits, and compares Logistic Regression with a tuned, class-balanced LinearSVC. Confusion matrices and classification reports are used to inspect routing errors.

### 4. Retrieval-augmented generation

The same Bitext data becomes the knowledge base. Each customer instruction is embedded with a Sentence-Transformer and indexed in FAISS using normalized vectors. For a new message, the retriever returns the most similar support examples and their paired responses, intents, and categories. A Groq-hosted LLM receives the user message, detected signals, and retrieved context through a prompt template, which keeps the answer tied to the support knowledge base. Retrieval quality is evaluated with hit@k and mean reciprocal rank on held-out examples.

## Data flow and reuse

The notebook keeps preprocessing task-specific:

- Language identification uses minimal, script-preserving cleaning.
- Emotion classification keeps a lightly cleaned BERT input and a more heavily cleaned classical-model input.
- Intent classification normalizes support messages for sparse linear models.
- RAG retains the original instruction and response text so retrieved context remains readable.

Shared configuration dataclasses, metric functions, random seeds, and a reusable transformer fine-tuning helper reduce duplication across the classification experiments.

## Running the notebook

The notebook is designed for Google Colab, Kaggle, or another Python environment with access to Hugging Face datasets and model downloads. It installs or expects the following main packages:

```text
datasets transformers evaluate accelerate emoji
torch scikit-learn pandas numpy matplotlib seaborn
sentence-transformers faiss-cpu groq
```

Run cells from top to bottom. Dataset downloads, transformer fine-tuning, and embedding/index construction require network access and can use substantial CPU/GPU memory. Before running the RAG generation section, provide a Groq API key through the `GROQ_API_KEY` environment variable; do not commit the key to the repository.

## Reproducibility and limitations

- Random seeds are set in the notebook for the train/validation/test splits and model experiments.
- The Bitext dataset has one supplied split, so the notebook creates stratified evaluation splits locally.
- Emotion labels come from Twitter-style text, so support-domain performance may differ from the benchmark test score.
- The generated answer is only as reliable as the retrieved examples and the configured LLM. Production use should add authentication, rate limits, logging controls, prompt-injection defenses, and human escalation.
- The notebook contains exploratory training and demo code rather than a deployed API or frontend service.

## Repository contents

- [ahmadtaher_nlp_final_task_rest_of_modules.ipynb](ahmadtaher_nlp_final_task_rest_of_modules.ipynb): dataset preparation, experiments, evaluation, retrieval, and end-to-end demos.
- [README.md](README.md): project architecture, data flow, setup, and operational notes.