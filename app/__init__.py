from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['IMAGE_EXTS'] = [".png", ".jpg", ".jpeg", ".gif", ".tiff"]
    from .routes import main
    app.register_blueprint(main)

    return app
