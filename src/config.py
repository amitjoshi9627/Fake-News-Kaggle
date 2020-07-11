import transformers
import os

NUM_LABELS = 3
OUTPUT_ATTENTIONS = False
OUTPUT_HIDDEN_STATES = True

DIR_ROOT = os.getcwd()
TRAIN_DATA_PATH = os.path.join(DIR_ROOT,"../data/train.csv")
TEST_DATA_PATH = os.path.join(DIR_ROOT,"../data/test.csv")
SUBMISSION_PATH = os.path.join("../data/sample_submission.csv")

BERT_PATH = os.path.join(DIR_ROOT,"../Bert_base_uncased")

BERT_DOWNLOAD_PATH = 'bert-base-uncased'

FEATURES = ['title1_en','title2_en']
TARGET = ['label']
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH)
MODEL = transformers.BertForSequenceClassification.from_pretrained(BERT_PATH,num_labels = NUM_LABELS, output_attentions = OUTPUT_ATTENTIONS, output_hidden_states = OUTPUT_HIDDEN_STATES)
MODEL_PATH = os.path.join(DIR_ROOT,"../Saved_Model/final_svm_model.pkl")

MAX_LEN = 64
BATCH_SIZE = 1