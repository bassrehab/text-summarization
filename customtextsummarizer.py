class CustomTextSummarizer(BaseTextSummarizer):
    def __init__ (self, model_key, language):
        self._tokenizer = AutoTokenizer.from_pretrained(model_key)

        self._language = language

        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_key)

        self._device = 'cuda' if bool(strtobool(os.getenv('USE_GPU'))) else 'cpu'

    def __get_chunk_text(self, text):
        sentences = [ s + ' ' for s in sentence_segmentation(text, minimum_n_words_to_accept_sentence=1, language=self._language) ]

        chunks = []

        chunk = ''

        length = 0

        for sentence in sentences:
            tokenized_sentence = self._tokenizer.encode(sentence, truncation=False, max_length=None, return_tensors='pt') [0]

            if len(tokenized_sentence) > self._tokenizer.model_max_length:
                continue

            length += len(tokenized_sentence)

            if length <= self._tokenizer.model_max_length:
                chunk = chunk + sentence
            else:
                chunks.append(chunk.strip())
                chunk = sentence
                length = len(tokenized_sentence)

        if len(chunk) > 0:
            chunks.append(chunk.strip())

        return chunks

    def __get_clean_text(self, text):
      if text.count('.') == 0:
        return text.strip()

      end_index = text.rindex('.') + 1

      return text[0 : end_index].strip()

    def summarize(self, text, *args, **kwargs):
        chunk_texts = self.__get_chunk_text(text)

        chunk_summaries = []

        for chunk_text in chunk_texts:
            input_tokenized = self._tokenizer.encode(chunk_text, return_tensors='pt')

            input_tokenized = input_tokenized.to(self._device)

            summary_ids = self._model.to(self._device).generate(input_tokenized, length_penalty=3.0, min_length = int(0.2 * len(chunk_text)), max_length = int(0.3 * len(chunk_text)), early_stopping=True, num_beams=5, no_repeat_ngram_size=2)

            output = [self._tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]

            chunk_summaries.append(output)

        summaries = [ self.__get_clean_text(text) for chunk_summary in chunk_summaries for text in chunk_summary ]

        return summaries
