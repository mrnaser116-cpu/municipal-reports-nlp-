Benchmarks

Municipal Reports – NLP Capstone Project

1. Tokenization – Lab 1

Metric	Result
Average Fertility	3.064
P90 Sequence Length	32
UNK Rate	0.0

These results are recorded in the original notebook.

2. Attention Tests – Lab 2

Metric	Result
Attention Tests	5
Passed Tests	5
Failed Tests	0

Manual scaled dot-product attention was compared with the PyTorch implementation using synthetic inputs.

3. Topic Classification – Lab 3A

Metric	Result
Total Reports	12
Topic Categories	5
Training Samples	8
Testing Samples	4
Shared Citizen IDs	0
Training Epochs	5
Batch Size	4
Learning Rate	3e-5
Evaluation Loss	2.24567

Classification accuracy and Macro-F1 were not recorded in the original training output.

4. NER Alignment – Lab 3B

Metric	Result
Word Alignment Tests	8
Passed Tests	8

These results measure word alignment coverage, not trained NER accuracy.

5. Arabic Dialect Processing – Lab 4

Model	Illustrative Macro-F1
CAMeLBERT-mix	0.92
CAMeLBERT-DA	0.99

These values are manually defined educational examples, not measured model performance.

6. Semantic Retrieval – Lab 5

Metric	Result
FAISS Indexed Reports	3
Recall@10	1.0
Original Notebook MRR@10	0.25
Standard Reciprocal Rank	0.50

The original notebook uses a non-standard rank offset. The relevant report is second in the manually defined ranking, giving a standard reciprocal rank of 0.50. The original code remains unchanged.

7. Evaluation – Lab 6

Evaluation examples cover language, dialect, and report length. Manually defined outcomes are used for educational demonstrations rather than validated model performance.

8. Deployment Benchmarks

* Streamlit application: deployed.
* Desktop and mobile access: manually tested.
* Production latency: not measured.
* Throughput: not measured.
* Concurrent request performance: not measured.

Live Demo

https://baladiya-nlp-demo.streamlit.app

Conclusion

The recorded results demonstrate educational NLP experiments. Further testing is required before making production-level performance claims.
