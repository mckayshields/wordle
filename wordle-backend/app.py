from flask import Flask, request, jsonify
import numpy as np
from information_theory import compute_highest_entropy, filter_words, load_words

app = Flask(__name__)

# Load all necessary data
allowed_guesses = load_words("allowed_guesses.txt")
possible_words = load_words("possible_secret_words.txt")
all_guess_results = np.load("all_guess_results.npy")

# State to persist filtered words/results between calls
current_possible_words = possible_words.copy()
current_guess_results = all_guess_results.copy()


@app.route("/next-guess", methods=["POST"])
def next_guess():
    global current_possible_words, current_guess_results

    data = request.get_json()
    guess = data.get("guess")
    result = data.get("result")

    if not guess or len(guess) != 5 or guess not in allowed_guesses:
        return jsonify({"error": "Invalid guess"}), 400
    if not isinstance(result, list) or len(result) != 5:
        return jsonify({"error": "Invalid result"}), 400

    # Filter words based on feedback
    current_possible_words, current_guess_results = filter_words(
        current_guess_results, allowed_guesses, current_possible_words, guess, result
    )

    if len(current_possible_words) == 1:
        next_word = current_possible_words[0]
    else:
        next_word = compute_highest_entropy(current_guess_results, allowed_guesses)

    return jsonify({"next_guess": next_word})


@app.route("/reset", methods=["POST"])
def reset():
    global current_possible_words, current_guess_results
    current_possible_words = possible_words.copy()
    current_guess_results = all_guess_results.copy()
    return jsonify({"message": "Game state reset."})


if __name__ == "__main__":
    app.run(debug=True)
