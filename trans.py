from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import json
import os
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):

    """
    Initialize the model and tokenizer for inference.
    """

    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        # low_cpu_mem_usage=True,
        quantization_config=qconfig
    )

    if qconfig==None:
        model = model.to(DEVICE)
        model.half()

    model.eval()

    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):

    """
    Inference on a batch of sentences.
    """

    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(), src=False
        )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(
            generated_tokens, lang=tgt_lang
        )

        del inputs
        torch.cuda.empty_cache()

    return translations

def combine_sentences(sentences, max_length=700):

    """
    sentences: List of sentences
    new_list: List of sentences where each sentence is less than max_length

    If sequence length in a batch is small, we combine multiple sentences into one while ensuring that the length of the combined sentence is less than max_length.
    The max_length is set to 700 characters, assuming it to be safe proxy for max_length=256 tokens for generation.
    This is done, so that by having comparable sequence lengths in a batch, we can use same batch size, achieving better GPU utilization.
    """

    new_list = []
    current_sentence = ""

    for sentence in sentences:
        if len(current_sentence) + len(sentence) <= max_length:
            current_sentence += sentence
        else:
            new_list.append(current_sentence)
            current_sentence = sentence

    # Add the last sentence, if any
    if current_sentence:
        new_list.append(current_sentence)

    return new_list

def chunk_and_translate(batched_data, en_indic_model, en_indic_tokenizer, ip):
  
  """
    Chunk the batched data into smaller chunks and translate them.
    Keep track of the sentence breaks (".") and new line characters via indices.
    Join the translated chunks back using the indices.
  """

  idx_list = []
  minibatch = []
  consecutive_newline_info = {}

  for each_row in batched_data:
      rows_split_by_newline = each_row.split("\n")

      for line in rows_split_by_newline:
          if line.strip():
              line_split = [k.strip() for k in sent_tokenize(line) if k.strip()]
              line_split = combine_sentences(line_split)
              minibatch.extend(line_split)
              consecutive_newline_info[len(minibatch)] =  1
          else:
            if len(minibatch) in consecutive_newline_info:
                consecutive_newline_info[len(minibatch)] += 1
            else:
                consecutive_newline_info[len(minibatch)] = 1

      idx_list.append(len(minibatch))

  translations = batch_translate(minibatch, "eng_Latn", "hin_Deva", en_indic_model, en_indic_tokenizer, ip)

  row_data = []
  start_idx = 0
  for idx in idx_list:
      translation_segment = translations[start_idx:idx]

      insert_count = 0
      for pos, count in consecutive_newline_info.items():
          if start_idx <= pos < idx:
              adjusted_index = pos + insert_count - start_idx
              translation_segment.insert(adjusted_index, "\n" * count)
              insert_count += 1 # No matter how many new line character are added, we are adding only one element in translation_segment list and therefore we put 1 here

      translated_text = " ".join(translation_segment).replace(" \n", "\n").replace("\n ", "\n")
      row_data.append(translated_text)
      start_idx = idx

  return row_data

def process(dataset, text_column, input_file_path, en_indic_model, en_indic_tokenizer, ip):
    """
    Preprocess the dataset and translate it.
    Modify this as per your needs.
    """

    total_batches = len(dataset) // ROW_BATCH + (1 if len(dataset) % ROW_BATCH != 0 else 0)

    if not os.path.exists("save"):
        os.makedirs("save")

    output_file_path = os.path.join("save", os.path.basename(input_file_path).replace(".jsonl", "_translated.jsonl"))

    for i in tqdm(range(0, len(dataset), ROW_BATCH), total=total_batches, desc="Processing Batches"):

        batched_data = dataset[i:min(i + ROW_BATCH, len(dataset))]
        batched_data = [k[text_column] for k in batched_data] 
        translated_rows = chunk_and_translate(batched_data, en_indic_model, en_indic_tokenizer, ip)

        with open(output_file_path, 'a', encoding='utf-8') as file:
           for rowidx, row in enumerate(translated_rows):
              file.write(json.dumps({"idx": rowidx, "text": row}, ensure_ascii=False) + "\n")

if __name__ == "__main__":

    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B" # ai4bharat/indictrans2-en-indic-dist-200M
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 12 # Modify based upon GPU memory
    ROW_BATCH = BATCH_SIZE*10 # Concatentate BATCH_SIZE*10 rows and then process them in chunks of BATCH_SIZE
    quantization = ""
    text_column = "text" # Text column name in your dataset that you want to translate

    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", quantization)
    ip = IndicProcessor(inference=True)
    
    input_file_path = sys.argv[1]
    with open(input_file_path, 'r') as file:
        dataset = [json.loads(line) for line in file]
        
    process(dataset, input_file_path, en_indic_model, en_indic_tokenizer, ip)