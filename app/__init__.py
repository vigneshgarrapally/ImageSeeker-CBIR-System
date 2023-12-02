from flask import Flask

def create_app():
    app = Flask(__name__)
    app.secret_key = 'a1b2c3d4e5f6g7h8i9j0'
    app.config['IMAGE_EXTS'] = [".png", ".jpg", ".heic"]
    from .routes import main
    app.register_blueprint(main)

    return app
