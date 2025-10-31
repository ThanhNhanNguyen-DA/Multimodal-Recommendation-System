from flask import Flask, request, jsonify
from recommender import Recommender

app = Flask(__name__)
recommender = Recommender()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    text_input = data.get('text')
    image_input = data.get('image')  # Assuming image is sent as base64 or URL

    recommendations = recommender.get_recommendations(text_input, image_input)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)