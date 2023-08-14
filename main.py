import os
import numpy as np
import pickle
from PIL import Image
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import tensorflow
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from io import BytesIO
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load pickle files and set up model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

@app.route('/get_similar_images', methods=['POST'])
def get_similar_images():
    try:
        image_url = request.json.get('imageUrl')  # Get imageUrl from JSON data

        if image_url:
            # Fetch the input image and calculate its features
            response = requests.get(image_url)
            response.raise_for_status()  # Raise exception for invalid URLs
            img = Image.open(BytesIO(response.content))

            input_features = feature_extraction(img, model)

            # Fetch all product documents from the Firestore "products" collection
            product_ref = db.collection('products')
            product_docs = list(product_ref.stream())  # Convert generator to a list

            # Calculate cosine similarity between input image and all images in the database
            similarity_scores = []
            for product_doc in product_docs:
                product_data = product_doc.to_dict()
                try:
                    db_feature = feature_extraction_from_db(product_data['imageUrl'], model)
                    similarity = cosine_similarity([input_features], [db_feature])
                    similarity_scores.append(similarity)
                except requests.exceptions.HTTPError:
                    continue

            # Create a list of tuples with (product_data, similarity_score) for sorting
            product_similarity_pairs = zip(product_docs, similarity_scores)
            product_similarity_pairs = sorted(product_similarity_pairs, key=lambda x: x[1], reverse=True)

            similar_products = []

            for product_doc, similarity_score in product_similarity_pairs[:3]:  # Take only top 3
                product_data_dict = product_doc.to_dict()  # Convert DocumentSnapshot to dictionary
                similar_products.append({'product_data': product_data_dict, 'similarity_score': similarity_score[0][0]})

            # Extract product_data and create the response
            result = [{'product_data': product['product_data'], 'similarity_score': float(product['similarity_score'])} for product in similar_products]

            return jsonify(result)
        else:
            return jsonify({'error': 'No imageUrl provided'}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': "Internal Server Error"}), 500

def feature_extraction(img, model):
    img_resized = img.resize((224, 224))  # Resize the image to (224, 224)
    img_array = image.img_to_array(img_resized)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def feature_extraction_from_db(image_url, model):
    response = requests.get(image_url)
    response.raise_for_status()  # Raise exception for invalid URLs
    img = Image.open(BytesIO(response.content))
    return feature_extraction(img, model)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
