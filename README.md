# CBIR System for Digital Libraries

## Project Overview

ImageSeeker is a Flask-based web application designed for a robust Content-Based Image Retrieval (CBIR) system. It uses advanced machine learning models to search and retrieve similar images from a local database, providing an intuitive way to explore image galleries.

![Alt text](image.png)

## Features

- Sync images from Google Photos using gphotos-sync.
- Extract features from images using facebook/dinov2-small model.
- Index images using FAISS for efficient similarity search.
- Search functionality to find similar images.
- Gallery view to display indexed images.

## Technology Stack:

- Feature Extraction and Matching: OpenCV, Pytorch
- Indexing: Facebook AI Similarity Search
- Backend: Python (Flask)
- Frontend: HTML, CSS, JavaScript

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Flask
- OpenCV
- Pytorch
- Transformers
- FAISS (Facebook AI Similarity Search)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aniruth5510/ImageSeeker-CBIR-System.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ImageSeeker-CBIR-System/
   ```

3. Install required packages:

   ```bash
   conda env create -f faiss.yaml
   ```

### Running the Application

1. Start the Flask server:

   ```bash
   python run.py
   ```

2. Access the application in a web browser at `http://localhost:5000`.

#### First-time Setup

1. **OAuth Client ID:**
   - Follow the [Creating an OAuth Client ID](https://gilesknap.github.io/gphotos-sync/main/tutorials/oauth2.html#client-id) guide.
   - Save the `client_secret.json` in the appropriate configuration directory.

2. **Sync Images:**
   - Run `gphotos-sync .\gallery\` in the command line to sync images from Google Photos.

#### Usage

- **Home Page**: Access at `http://localhost:5000/`.
- **Setup Page**: Instructions for setting up OAuth Client ID.
- **Sync Page**: Sync images and update the FAISS index.
- **Gallery Page**: View the synced image gallery.
- **Search Page**: Upload an image and specify the number of similar images to retrieve.

#### Updating the FAISS Index

- Images are automatically added to the FAISS index during the sync process.
- The index is updated with new images, and deleted images are removed to keep the index current.
