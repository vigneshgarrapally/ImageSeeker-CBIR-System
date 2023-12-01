"""
This module contains the routes for the Flask application.

Attributes:
    main (Blueprint): The main Blueprint for the Flask application.
"""

from pathlib import Path
import binascii
from flask import Blueprint, render_template, current_app, send_from_directory


main = Blueprint('main', __name__)

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
    # Logic for the search page
    return render_template('search.html')
