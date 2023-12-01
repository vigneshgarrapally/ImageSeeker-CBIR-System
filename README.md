# CBIR System for Digital Libraries

## Project Overview

This Content-Based Image Retrieval (CBIR) System is designed for digital image galleries, enabling users to search and retrieve images based on content similarity. The system allows users to upload an image as a query, which is then processed to find and display similar images from the database.

## Features

- **Image Upload and Preprocessing**: Users can upload query images in various formats and from online services like Google Photos.
- **Content-Based Image Search**: The system uses deep learning or traditional computer vision methods to extract features from the uploaded image and compares them with images in the database to identify similar images.
- **Metadata Display**: Displays metadata such as title, date, and source alongside the images for better context.
- **Technology Stack**:
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
- Facebook AI Similarity Search

### Installation

1. Clone the repository:

   ```bash
   git clone [repository URL]
   ```

2. Navigate to the project directory:

   ```bash
   cd [project directory]
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Flask server:

   ```bash
   python app.py
   ```

2. Access the application in a web browser at `http://localhost:5000`.

## Usage

- Upload an image using the provided interface.
- The system will process the image and display similar images from the database along with their metadata.

## Acknowledgments

- OpenCV contributors
- Pytorch community
- Facebook AI Similarity Search team
