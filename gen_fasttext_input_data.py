import os
import math
import sys
import glob
import json
import re
import pickle
DATA_SIZE = 300000
def _make_char_index():
  char_index = {}
  for char in open('./char_level.txt', 'r').read().replace('\n', ' ').split():
    emoji = re.compile(u'['
          u'\U0001F300-\U0001F5FF'
          u'\U0001F600-\U0001F64F'
          u'\U0001F680-\U0001F6FF'
          u'\u2600-\u26FF\u2700-\u27BF]+', 
          re.UNICODE)
    if re.match(emoji, char) is not None:
      continue
    if char_index.get(char) is not None:
      continue
    char_index[char] = len(char_index)
    print(char)
  open('char_index.pkl', 'wb').write(pickle.dumps(char_index))
  sys.exit()
def make_char():
  os.system('rm -rf ./dataset')
  os.system('mkdir dataset')
  f = open('char_level.txt', 'w')
  for ni, name in enumerate(glob.glob('../out20170325/*')):
    if ni%10000 == 0:
      print("iter %d"%ni, file=sys.stderr)
    if ni > DATA_SIZE : break
    try:
      obj = json.loads(open(name).read())
    except:
      continue
    text = obj['txt']
    emoji = re.compile(u'['
            u'\U0001F300-\U0001F5FF'
            u'\U0001F600-\U0001F64F'
            u'\U0001F680-\U0001F6FF'
            u'\u2600-\u26FF\u2700-\u27BF]+', 
            re.UNICODE)
    if re.search(emoji, text) is None:
      continue
    emojis = re.findall(emoji, text)
    for p in [r' ', r'\n', r'　', emoji]:
      no_emoji = re.sub(p, '', text)
    emojis.extend(list(no_emoji))
    f.write("%s\n"%' '.join(emojis))
  # ./fasttext skipgram -input char_level.txt -output model -dim 256 -minCount 1
  os.system("./fasttext skipgram -input char_level.txt -output model -dim 256 -minCount 1")

def make_term_vec():
  with open('./model.vec', 'r') as f:
    next(f)
    term_vec = {}
    for fi, line in enumerate(f):
      if fi%1000 == 0:
        print("iter %d"%fi)
      line = line.strip()
      ents = line.split()
      term = ents.pop(0)
      vec  = [float(v) for v in ents]
      term_vec[term] = vec
    open('term_vec.pkl', 'wb').write(pickle.dumps(term_vec))

def make_dataset():
  term_vec = pickle.loads(open('term_vec.pkl', 'rb').read())
  for ni, name in enumerate(glob.glob('../out20170325/*')):
    target_name = name.split('/')[-1]
    if ni%10000 == 0:
      print("iter %d"%ni, file=sys.stderr)
    if ni > DATA_SIZE : break
    try:
      obj = json.loads(open(name).read())
    except:
      continue
    text = obj['txt']
    emoji = re.compile(u'['
            u'\U0001F300-\U0001F5FF'
            u'\U0001F600-\U0001F64F'
            u'\U0001F680-\U0001F6FF'
            u'\u2600-\u26FF\u2700-\u27BF]+', 
            re.UNICODE)
    if re.search(emoji, text) is None:
      continue
    emojis = re.findall(emoji, text)
    try:
      emojis_vec = [term_vec[e] for e in emojis]
    except KeyError as e:
      continue
    no_emoji = text
    for p in [r'\s', r'\n', r'　', emoji]:
      no_emoji = re.sub(p, '', no_emoji)
    out_text = ['*']*10
    out_text.extend(list(no_emoji))
    for slicer in range(0, len(out_text) - 11, 1):
      try:
        input_vec = [term_vec[w] for w in out_text[slicer:slicer+10]]
      except KeyError as e:
        continue
      ans = out_text[slicer+10]
      open('./dataset/%s.slicer%d.pkl'%(target_name, slicer), 'wb').write(pickle.dumps([input_vec, "".join(emojis), emojis_vec, ans]))
      #print(''.join(out_text))
if __name__ == '__main__':
  print(len([f for f in glob.glob('./dataset/*')]))
  if '--all' in sys.argv:
    make_char()
    make_term_vec()
    make_dataset()
  if '--make_char' in sys.argv:
    make_char()
  if '--make_term_vec' in sys.argv:
    make_term_vec()
  if '--make_dataset' in sys.argv:
    make_dataset()
