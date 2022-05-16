import sys
from flask import Flask
from flask import jsonify
from flask import render_template




# importing sys

# adding Folder_2 to the system path
sys.path.insert(0, '/Users/ian/Code/JS/HR/MVP/server/venv/funcs')
from create import create

app = Flask(__name__)

@app.route("/")
def serve():
    #return "<p>Hello, World!</p>"
    return render_template('index.html')


@app.route("/api/create/", methods=['GET'])
def users():
    newData=create()
    #print(newData[1])
    return  jsonify(newData)
