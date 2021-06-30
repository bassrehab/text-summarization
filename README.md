# text-summarization

This is a quick experiment to test extractive and abstractive text summarization with the SOTA Transformer based pre-trained models of Google T5, Google Pegasus, Facebook BART CNN.
Data Sources used as input 0 include publicly available PDFs in the data folder (/data/Accenture*.pdf, and  /data/BSR*.pdf)
Outputs of summaries are available in the Jupyter Notebook and /data/summary*.txt
Files named /data/extract*.txt are full text extracts from the PDF sources.

These experiments were run on GCP, with tensorflow 2.3.3 and Pytorch 1.4.0.