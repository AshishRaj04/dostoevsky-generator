import pickle
from engine.tokenization import Tokenizer
with open("./engine/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
vocab = str(tokenizer.vocab)  

with open("./assets/vocab.txt", "w") as f:
    f.write(vocab)