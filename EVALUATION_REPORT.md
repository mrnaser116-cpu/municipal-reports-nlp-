Evaluation Report

Municipal Reports – NLP Capstone Project

Evaluation Scope

The project is an educational prototype for Arabic municipal report analysis using NLP and Transformers.

Lab 1 – Text Preprocessing and Tokenization

* Text cleaning and whitespace normalization.
* Personal information masking using regular expressions.
* Tokenization and fertility analysis.

Lab 2 – Transformer Architecture

* Scaled dot-product attention.
* Multi-head attention and attention visualization.
* Comparison with PyTorch attention implementation.

Lab 3A – Topic Classification

* 12 Arabic municipal reports.
* Five topic categories.
* Eight training samples and four testing samples.
* No shared citizen IDs between training and testing sets.
* Live classification is not currently enabled in the demo.

Lab 3B – Named Entity Recognition

* BIO tagging for ISSUE and LOC entities.
* Token and word alignment.
* Manual evaluation examples, not trained NER model results.

Lab 4 – Arabic Dialect Processing

* Arabic text normalization.
* Saudi dialect examples.
* Illustrative model comparison values.

Lab 5 – Semantic Retrieval

* Multilingual sentence embeddings.
* FAISS similarity search.
* Cross-encoder relevance scoring.
* Educational retrieval evaluation.

Lab 6 – Evaluation and Error Analysis

* Evaluation by language, dialect, and report length.
* Illustrative evaluation samples.
* Bootstrap confidence interval demonstration.

Limitations

* Small educational dataset.
* Some evaluation values are manually defined.
* Production-level accuracy and latency have not been validated.
* Further training and evaluation are required.

Live Demo

https://baladiya-nlp-demo.streamlit.app

Conclusion

The project demonstrates the NLP concepts covered in Labs 1–6 through an interactive Streamlit application while preserving the original laboratory notebook.
