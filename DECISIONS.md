Technical Decisions

Municipal Reports – NLP Capstone Project

1. Python

Python is used for text processing, model experiments, data analysis, and application development.

2. Regular Expressions (Regex)

Regular expressions provide a simple approach to text cleaning and masking selected personal information, including phone numbers, email addresses, and report identifiers.

3. CAMeLBERT

CAMeLBERT is suitable for exploring Arabic NLP tasks because it is pretrained on Arabic text.

4. PyTorch

PyTorch supports Transformer experiments, attention calculations, and neural network training.

5. Hugging Face Transformers

Hugging Face provides pretrained models and tokenizers for Arabic and multilingual NLP experiments.

6. Sentence Transformers

Multilingual sentence embeddings enable semantic similarity comparisons between municipal reports.

7. FAISS

FAISS provides efficient vector similarity search for retrieving related municipal reports.

8. Cross-Encoder

A cross-encoder provides relevance scores for query and report pairs.

9. Group-Based Data Splitting

Grouping by citizen ID prevents the same citizen from appearing in both training and testing sets.

10. Streamlit

Streamlit provides an interactive web interface for demonstrating the laboratory experiments without requiring complex frontend development.

11. GitHub

GitHub is used for source code management, project documentation, and access to the original laboratory notebook.

12. Project Scope

* Preserve the original laboratory notebook.
* Provide an interactive educational demonstration.
* Distinguish illustrative results from measured performance.
* Avoid unsupported production-level performance claims.
* Identify limitations and opportunities for future improvement.

Live Demo

https://baladiya-nlp-demo.streamlit.app

Conclusion

The selected technologies support an educational prototype for municipal report analysis using Arabic NLP and Transformers.
