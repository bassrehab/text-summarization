
# Experiments on Text Summarization - SOTA Transformers

This is a quick experiment to test **extractive** and **abstractive** text summarisation with the SOTA Transformer based pre-trained models of 

 - [Google T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) ,  
 - [Google Pegasus](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html),  
 - [Facebook BART CNN](https://ai.facebook.com/research/publications/bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation-translation-and-comprehension/)

## Input/Output Files
Data Sources used as input include publicly available PDFs in the data folder 
 - */data/Accenture\*.pdf*,  
 - */data/BSR\*.pdf*

Outputs of summaries are available in 

 - the Jupyter Notebook  
 - */data/summary\*.txt*

Files named */data/extract*.txt* are full text extracts from the PDF sources.

## Summaries
Two type of summaries were tested.

 - **Full PDF summaries:** Full PDF was extracted and summarized as a whole.
 - **Pagewise summaries:** PDF pages were extracted and summarised.

## Environment Details

- These experiments were run on GCP (without GPU), with tensorflow 2.3.3 and Pytorch 1.4.0.
 - Environment: TensorFlow Enterprise 2.3 (with LTS and Intel®   
   MKL-DNN/MKL)
- **Machine type**: e2-highmem-16 (Efficient Instance, 16   
   vCPUs, 128 GB RAM)   
- **GPU** : None
- accompanying *requirements.txt* has the full dump of the environment alongwith dependencies.


## Pre-processing
Limited pre-processing to remove whitespaces.

## To Do
Fine tune chunk size, truncations and custom tokenizer initialization.
