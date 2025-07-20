import tensorflow as tf
from engine.transformer import create_transformer
import pickle 
from flask import Flask, jsonify
from flask_cors import CORS
from engine.tokenization import Tokenizer
# tokenizer = Tokenizer()

context_length = 256

with open("engine/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
def load_parameters():
    global model
    model = create_transformer(vocab_size=1256, emb_dim=384, n_heads=8, Nx=6)
    dummy_input = tf.zeros((1, context_length), dtype=tf.int32)
    model(dummy_input)
    model.load_weights("./checkpoints/dostoevsky.weights.h5")

    print("Weights loaded successfully.")


def generate_samples(load=True, num_tokens=100):
    generated = []
    context = [0] * context_length

    if load:
        load_parameters()
        for _ in range(num_tokens):
            Xb = tf.constant([context], dtype=tf.int32)
            logits = model(Xb, training=False)
            last_logits = logits[:, -1, :]
            probs = tf.nn.softmax(last_logits, axis=-1)
            ix = tf.random.categorical(tf.math.log(probs), num_samples=1)[0, 0].numpy()
            context = context[1:] + [ix]
            generated.append(ix)
        print("Sequence Generated")
        # generated = [int(x) for x in generated]
        return generated
    else:
        raise ValueError("Model parameters not loaded. Please load the model first.")


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Text Generation API"}), 200


@app.route("/api/generate", methods=["POST"])
def generate_text():
    try:
        num_tokens = 100

        generated_tokens = generate_samples(load=True, num_tokens=num_tokens)
        generated_text = tokenizer.decode(generated_tokens)
        # if not isinstance(generated_text, str):
        #     generated_text = str(generated_text)
        return jsonify({"status": "success", "generated_text": generated_text})
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Error in calling api/generate : {str(e)}"}
        ), 500


if __name__ == "__main__":
    print("Starting the Flask app.....")
    app.run(debug=True, port=8000)

