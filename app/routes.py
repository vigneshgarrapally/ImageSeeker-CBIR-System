"""
This module contains the routes for the Flask application.

Attributes:
    main (Blueprint): The main Blueprint for the Flask application.
"""

from pathlib import Path
import binascii
import os
import pickle
from flask import Blueprint, render_template, current_app, send_from_directory,flash,redirect,url_for
import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from flask import request, jsonify
from werkzeug.utils import secure_filename
from pillow_heif import register_heif_opener

register_heif_opener()
main = Blueprint('main', __name__)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
model.eval()


@main.route('/')
def index():
    """
    Renders the index.html template.

    Returns:
        The rendered index.html template.
    """
    return render_template('index.html')

@main.route('/setup')
def setup():
    # Logic for Google account setup
    return render_template('setup.html')

@main.route('/sync')
def sync():
    # Logic for syncing latest images
    return render_template('sync.html')

def encode(filepath):
    """
    Encodes the given filepath into a hexadecimal string.

    Args:
        filepath (str): The path of the file to be encoded.

    Returns:
        str: The hexadecimal representation of the filepath.
    """
    return binascii.hexlify(filepath.encode('utf-8')).decode()

@main.route('/cdn/<path:filepath>')
def download_file(filepath):
    """
    Download a file from the specified filepath.

    Args:
        filepath (str): The path of the file to be downloaded.

    Returns:
        str: The file content if found, or "File not found" with status code 404.
    """
    filepath_decoded = binascii.unhexlify(filepath.encode('utf-8')).decode()
    gallery_root = Path(__file__).resolve().parent.parent / 'gallery'
    file_path = gallery_root / filepath_decoded
    if not file_path.is_file():
        return "File not found", 404
    return send_from_directory(str(file_path.parent), file_path.name, as_attachment=False)


@main.route('/gallery')
def gallery():
    """
    Renders the gallery page with a list of image paths.

    Returns:
        A rendered HTML template with the image paths.
    """
    root_dir = Path(__file__).resolve().parent.parent / 'gallery'
    image_paths = []
    for path in root_dir.glob('**/*'):  # Using glob instead of rglob if you want non-recursive
        if path.is_file() and path.suffix.lower() in [ext.lower() for ext in current_app.config['IMAGE_EXTS']]:
            rel_path = path.relative_to(root_dir)
            image_paths.append(encode(str(rel_path)))
    return render_template('gallery.html', paths=image_paths)

@main.route('/search')
def search():
    return render_template('search.html')


def perform_search(image_path, num_results):

    index = faiss.read_index('vector.index')
    image_list = pickle.load(open('images.pkl', 'rb'))

    # Load the model and processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    model.eval()

    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Normalize the features
    embeddings = outputs.last_hidden_state.mean(dim=1)
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)

    # Read the index and perform the search
    
    distances, indices = index.search(vector, num_results)
    encoded_paths = [encode(image_list[idx]) for idx in indices[0]]
    distances = distances[0].tolist()
    
    return list(zip(encoded_paths, distances))

@main.route('/perform_search', methods=['POST'])
def search_route():
    # Check if the index files exist
    if not os.path.exists('vector.index') or not os.path.exists('images.pkl'):
        flash('Please sync your images before searching.')
        return redirect(url_for('main.index'))

    # Process the uploaded image
    image_file = request.files['image']
    num_results = request.form.get('num_results', 3, type=int)
    if image_file:
        filename = secure_filename(image_file.filename)
        image_path = os.path.join('query_images', filename)
        #create query_images folder if it does not exist
        if not os.path.exists('query_images'):
            os.makedirs('query_images')
        image_file.save(image_path)

        # Perform the search
        search_results = perform_search(image_path, num_results)
        print(search_results)
        return render_template('results.html', search_results=search_results)