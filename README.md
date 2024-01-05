# TransHi   
English to Hindi Translation &amp; Hindi to Roman Hindi Transliteration   

## Installation IndicTrans2   
```
pip install -q -U transformers[sentencepiece] datasets

pip install -q nltk sacremoses regex pandas mock transformers==4.28.1 sacrebleu==2.3.1 urduhack[tf] mosestokenizer ctranslate2==3.9.0 gradio

git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
export INDIC_RESOURCES_PATH="/indic_nlp_resources"

git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
cd indic_nlp_library
pip install ./

import urduhack
urduhack.download()
import nltk
nltk.download('punkt')

git clone https://github.com/AI4Bharat/IndicTrans2.git
git clone https://github.com/VarunGumma/IndicTransTokenizer
cd IndicTransTokenizer
pip install --editable ./
```

## English to Hindi Translation   
1. Downloading dataset from HF `python prepare_data.py`   
2. Running translation with IndicTrans2 `python trans.py <input-file.jsonl>`   