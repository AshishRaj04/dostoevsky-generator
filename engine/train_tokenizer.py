import pickle

from tokenization import Tokenizer

path = "../data/test.txt"

with open(path , "r" , encoding="utf-8") as file:
    text = file.read()
    
tokenizer = Tokenizer()
tokenizer.train(text, vocab_size=512, verbose=True)
with open("../tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)