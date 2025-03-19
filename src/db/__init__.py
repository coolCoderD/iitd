from flask_pymongo import PyMongo
from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

mongo = PyMongo()

def init_db(app: Flask):
    # Set up MongoDB URI (replace with your connection string)
    app.config["MONGO_URI"] = os.getenv("MONGODB_URI")
    mongo.init_app(app)
    print("Connected to MongoDB!")
