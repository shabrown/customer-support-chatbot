import json
import re
import os
import unicodedata
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
import boto3

s3 = boto3.resource('s3')

encoder_dir = '/tmp/encoder'
decoder_dir = '/tmp/decoder'

encoder_var_dir = '/tmp/encoder/variables'
decoder_var_dir = '/tmp/decoder/variables'

all_dirs = [encoder_dir, decoder_dir, encoder_var_dir, decoder_var_dir]

for directory in all_dirs:
    if not os.path.exists(directory):
        os.mkdir(directory)

# Files that need to be downloaded for the encoder
encoder_bucket_name = 'attentionencoder-20'
encoder_pb_key = 'saved_model.pb'
variable_index_key = 'variables/variables.index'
variable_data_key = 'variables/variables.data-00000-of-00001'

s3.Bucket(encoder_bucket_name).download_file(encoder_pb_key, f"{encoder_dir}/saved_model.pb")
s3.Bucket(encoder_bucket_name).download_file(variable_index_key, f"{encoder_var_dir}/variables.index")
s3.Bucket(encoder_bucket_name).download_file(variable_data_key, f"{encoder_var_dir}/variables.data-00000-of-00001")

encoder = tf.saved_model.load(encoder_dir)


# Files that need to be downloaded for the encoder

decoder_bucket_name = 'attentiondecoder-20'
decoder_pb_key = 'saved_model.pb'
variable_index_key = 'variables/variables.index'
variable_data_key = 'variables/variables.data-00000-of-00001'

s3.Bucket(decoder_bucket_name).download_file(decoder_pb_key, f"{decoder_dir}/saved_model.pb")
s3.Bucket(decoder_bucket_name).download_file(variable_index_key, f"{decoder_var_dir}/variables.index")
s3.Bucket(decoder_bucket_name).download_file(variable_data_key, f"{decoder_var_dir}/variables.data-00000-of-00001")

# Load models
decoder = tf.saved_model.load(decoder_dir)


max_length_targ = 56
max_length_inp = 246
units = 1024


with open('target_index_word.json') as f:
    target_index_word = json.load(f)

with open('target_word_index.json') as f:
    target_word_index = json.load(f)

with open('input_word_index.json') as f:
    input_word_index = json.load(f)


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    """Pre-process users' requests or questions"""
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
               '<url>', w)
    # creating a space between a word and the punctuation following it
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿<>]+", " ", w)

    # lemmatize the word
    lemmatizer = WordNetLemmatizer()
    w = lemmatizer.lemmatize(w)

    w = w.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def respond(sentence):
    """Generate response using the model """
    # attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = sentence.lower()
    sentence = preprocess_sentence(sentence)

    inputs = []
    for i in sentence.split(' '):
        if i in input_word_index:
            inputs.append(input_word_index[i])
        else:
            pass

    inputs = keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    encoding_out, encoding_hidden = encoder.predict((inputs, hidden[0]))

    decoding_hidden = encoding_hidden
    decoding_input = tf.expand_dims([target_word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, decoding_hidden, attention_weights = decoder.decode((
            decoding_input, decoding_hidden, encoding_out))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += target_index_word[str(predicted_id)] + ' '

        if target_index_word[str(predicted_id)] == '<end>':
            return result

        # the predicted ID is fed back into the model
        decoding_input = tf.expand_dims([predicted_id], 0)

    return result


