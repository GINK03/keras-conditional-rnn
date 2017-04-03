from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.core import Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization as BN
from keras.callbacks import LearningRateScheduler as LRS
import numpy as np
import random
import sys
import glob
import pickle

start = 0.01
stop  = 0.001
nb_epoch = 50
learning_rates = np.linspace(start, stop, nb_epoch)

counter = 0
def build_model(mode=None, maxlen=None, output_dim=None):
  print('Build model...')
  def _scheduler(epoch):
    global counter
    counter += 1
    rate = learning_rates[counter]
    #model.lr.set_value(rate)
    print(model.optimizer.lr)
    #print(counter, rate)
    return model.optimizer.lr
  change_lr = LRS(_scheduler)
  model = Sequential()
  model.add(LSTM(128*15, return_sequences=False, input_shape=(maxlen, 512)))
  model.add(Dropout(0.5))
  model.add(Dense(output_dim))
  model.add(Activation('softmax'))
  if mode=="rms":
    optimizer = RMSprop(lr=0.01)
  if mode=="adam":
    optimizer = Adam()
  model.compile(loss='categorical_crossentropy', optimizer=optimizer) 
  return model, change_lr


def sample(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def dynast(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  return np.argmax(preds)


def train(mode):
  datasets   = []
  for ni, name in enumerate(glob.glob('./dataset/*')[:20000]):
    if ni%1000 == 0: 
      print("now loading pkl iter %d"%ni)
    try:
      with open(name, 'rb') as f:
        data = pickle.loads(f.read())
    except pickle.UnpicklingError as e:
      continue
    datasets.append(data)
  #datasets = datasets*100
  random.shuffle(datasets)
  max_texts = len(datasets)
  char_index = pickle.loads(open('./char_index.pkl', 'rb').read())
  index_char = {index:char for char, index in char_index.items()}
  term_vec   = pickle.loads(open('./term_vec.pkl', 'rb').read())
  X = np.zeros((max_texts, 10, 512), dtype=np.float64)
  Y = np.zeros((max_texts, len(char_index)), dtype=np.float64)
  emojis_vec = []
  for i, dataset in enumerate(datasets):
    if i%1000 == 0: 
      print("now mapping to numpy iter %d"%i)
    x, emoji, emoji_each_vec, y = dataset
    emoji_vec = np.zeros((256), dtype=np.float64)
    for e in set(list(emoji)):
      try:
        emoji_vec += np.array(term_vec[e])
      except KeyError as e:
        print("error %s"%e)
    emojis_vec.append( [emoji, emoji_vec] )
    Y[i, char_index[y]] = 1.
    for t, vec in enumerate(x):
      X[i, t,:256] = vec*emoji_vec
      X[i, t,256:] = vec + emoji_vec
  open('emojis_vec.pkl', 'wb').write(pickle.dumps(emojis_vec) )
  model, scheduler = build_model(mode=mode, maxlen=10, output_dim=len(char_index))
  for iteration in range(1, 10000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, Y, batch_size=64, nb_epoch=1)#, callbacks=[scheduler])
    MODEL_NAME = "./models.%s/%09d.model"%(mode, iteration)
    model.save(MODEL_NAME)
    if iteration%1==0:
      for diversity in [1.0]:
        print()
        print('----- diversity:', diversity)
        sent = ["*"]*10
        emoji_data = emojis_vec[random.randint(0, len(emojis_vec) - 1)]
        emoji, _emoji_vec = emoji_data
        print('----- Generating with seed: "' + emoji +" " + "".join(sent) + '"')
        sys.stdout.write("".join(sent))
        for i in range(200):
          x = np.zeros((1, 10, 512))
          try:
            for t, char in enumerate(sent):
              x[0, t, :256] = term_vec[char]*_emoji_vec
              x[0, t, 256:] = term_vec[char] + _emoji_vec
            preds = model.predict(x, verbose=0)[0]
            next_index = dynast(preds, diversity)
            next_char = index_char[next_index]
            sent.append(next_char)
            sent = sent[1:] 
            sys.stdout.write(next_char)
            sys.stdout.flush()
          except KeyError as e:
            break
        print()

def eval():
  INPUT_NAME = "./source/bocchan.txt"
  MODEL_NAME = "./models/%s.model"%(INPUT_NAME.split('/').pop())

  #path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
  #text = open(path).read().lower()
  text = open(INPUT_NAME).read()
  print('corpus length:', len(text))

  chars = sorted(list(set(text)))
  print('total chars:', len(chars))
  char_indices = dict((c, i) for i, c in enumerate(chars))
  indices_char = dict((i, c) for i, c in enumerate(chars))
  maxlen = 40
  step = 3
  sentences = []
  next_chars = []
  for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
  model = load_model(MODEL_NAME)

  #for diversity in [0.2, 0.5, 1.0, 1.2]:
  for diversity in [1.0, 1.2]:
    print()
    print('----- diversity:', diversity)
    generated = ''
    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
    #for i in range(400):
    for i in range(200):
      x = np.zeros((1, maxlen, len(chars)))
      for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.
      preds = model.predict(x, verbose=0)[0]
      #next_index = sample(preds, diversity)
      next_index = dynast(preds, diversity)
      next_char = indices_char[next_index]
      generated += next_char
      sentence = sentence[1:] + next_char
      sys.stdout.write(next_char)
      sys.stdout.flush()
    print()


def main():
  if '--train_adam' in sys.argv:
     train(mode="adam")
  if '--train_rms' in sys.argv:
     train(mode="rms")
  if '--eval' in sys.argv:
     eval()
if __name__ == '__main__':
  main()
