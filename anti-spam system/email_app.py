import numpy as np
import joblib
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# load  model and scaler at startup
xgb_model = joblib.load('spam_model_xgb.pkl')
scaler = joblib.load('spam_model_scaler.pkl')

def extract_features(email_text):
    """"Extract features from raw email text"""
    # Number of links (Counts UrL)
    num_links = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text))

    # number of words
    num_words = len(email_text.split())

    # has_offer (keywords like 'offer', 'discount', 'deal')
    has_offer = 1 if any(keyword in email_text.lower() for keyword in ['offer', 'discount', 'deal', 'free', 'click', 'won']) else 0

    # sender_score
    sender_score = 0.2

    # all_caps
    upper_char = sum(1 for c in email_text if c.isupper())
    total_char = len(email_text.replace(' ', ''))
    all_caps = 1 if total_char > 0 and (upper_char / total_char) >  0.5 else 0

    return {
        'num_links': num_links,
        'num_words': num_words,
        'has_offer': has_offer,
        'sender_score': sender_score,
        'all_caps': all_caps
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check if email_text is provided
        if 'email_text' in data:
            email_text = data['email_text']
            if not email_text:
                return  jsonify({'error': 'email_text is required'}), 400
            features = extract_features(email_text)

        else:
            # Except numeric features
            required_fields = ['num_links', 'num_words', 'has_offer', 'sender_score', 'all_caps']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required numerical features'}), 400
            features = {    
                'num_links': float(data['num_links']),
                'num_words': float(data['num_words']),
                'has_offer': int(data['has_offer']),
                'sender_score': float(data['sender_score']),
                'all_caps': int(data['all_caps'])
            }

        # Scale num_links and num_words
        scaled_values =scaler.transform([[features['num_links'], features['num_words']]])
        num_links_scaled = float(scaled_values[0][0])
        num_words_scaled = float(scaled_values[0][1])

         # Calculate links per_word
        links_per_word = float(num_links_scaled / num_words_scaled if num_words_scaled != 0 else 0.0)

        # Final features vector
        features_vector = np.array([[
             num_links_scaled,
             num_words_scaled,
             features['has_offer'],
             features['sender_score'],
             features['all_caps'], 
             links_per_word
            ]])

        # Predict
        proba = float(xgb_model.predict_proba(features_vector)[0][1])
        prediction = int(proba >= 0.42)

        return jsonify({
            'prediction': prediction,
            # 'spam_probability': round(proba, 3)
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'health'}), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)