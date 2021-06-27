import pdb

from transformers import BertTokenizer

from fastestimator.dataset.data import tednmt


def fastestimator_run():
    train_ds, eval_ds, test_ds = tednmt.load_data(translate_option="pt_to_en")
    pt_tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    pdb.set_trace()
    pt_tokenizer.tokenize(train_ds[0]["source"])
    data  = en_tokenizer.tokenize(train_ds[0]["target"])
    data2 = en_tokenizer.convert_tokens_to_ids(data)
    data3 = en_tokenizer(train_ds[0]["target"])
    en_tokenizer.convert_ids_to_tokens(data3["input_ids"])