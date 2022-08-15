from crypt import methods
from distutils.log import debug
from flask import Flask, session
import datetime
from flask import request
import os
from werkzeug.utils import secure_filename
from pathlib import Path

# DTOC IMPORTS
import networkx as nx
import pandas as pd
import datetime
import numpy as np
import math
from pathlib import Path

# Determine the current file path
file_path = Path(".").absolute()


# Create file path to create a folder and save results to said folder
UPLOAD_FOLDER = file_path/"temp"
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Members API Route


@app.route("/upload_csv", methods=['POST'])
def upload_csv():
    target = os.path.join(UPLOAD_FOLDER, 'uploads')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination = "/".join([target, filename])
    file.save(destination)
    # session['uploadFilePath'] = destination
    response = {"status": "success", "destination": destination}
    return response


if __name__ == "__main__":
    app.run(debug=True)
