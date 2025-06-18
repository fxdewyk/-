import os
import urllib.request
import ipfshttpclient

from my_constants import app
import pyAesCrypt
from flask import Flask, flash, request, redirect, render_template, url_for, jsonify
from flask_socketio import SocketIO, send, emit
from werkzeug.utils import secure_filename
import socket
import pickle
from blockchain import Blockchain
import requests

# The package requests is used in the 'hash_user_file' and 'retrieve_from hash' functions to send http post requests.
# Notice that 'requests' is different than the package 'request'.
# 'request' package is used in the 'add_file' function for multiple actions.

socketio = SocketIO(app)
blockchain = Blockchain()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def append_file_extension(uploaded_file, file_path):
    file_extension = uploaded_file.filename.rsplit('.', 1)[1].lower()
    user_file = open(file_path, 'a')
    user_file.write('\n' + file_extension)
    user_file.close()

def decrypt_file(file_path, file_key):
    encrypted_file = file_path + ".aes"
    os.rename(file_path, encrypted_file)
    pyAesCrypt.decryptFile(encrypted_file, file_path,  file_key, app.config['BUFFER_SIZE'])

def encrypt_file(file_path, file_key):
    pyAesCrypt.encryptFile(file_path, file_path + ".aes",  file_key, app.config['BUFFER_SIZE'])

# def hash_user_file(user_file, file_key):
#     encrypt_file(user_file, file_key)
#     encrypted_file_path = user_file + ".aes"
#
#     # client = ipfsapi.connect('127.0.0.1', 5001)
#     #这里我修改过：改成了本地的地址
#
#     client = ipfshttpclient.connect()
#
#     response = client.add(encrypted_file_path)
#     file_hash = response['Hash']
#     return file_hash


import requests


def hash_user_file(user_file, file_key):
    encrypt_file(user_file, file_key)
    encrypted_file_path = user_file + ".aes"

    # 使用 requests 上传到 IPFS
    url = 'http://127.0.0.1:5001/api/v0/add'
    with open(encrypted_file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        response.raise_for_status()
        file_hash = response.json()['Hash']

    return file_hash


# def retrieve_from_hash(file_hash, file_key):
#     # client = ipfsapi.connect('127.0.0.1', 5001)
#     #这里我也修改过，改成了本地的地
#     client = ipfshttpclient.connect()
#
#     file_content = client.cat(file_hash)
#     file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], file_hash)
#     user_file = open(file_path, 'ab+')
#     user_file.write(file_content)
#     user_file.close()
#     decrypt_file(file_path, file_key)
#     with open(file_path, 'rb') as f:
#         lines = f.read().splitlines()
#         last_line = lines[-1]
#     user_file.close()
#     file_extension = last_line
#     saved_file = file_path + '.' + file_extension.decode()
#     os.rename(file_path, saved_file)
#     print(saved_file)
#     return saved_file

def retrieve_from_hash(file_hash, file_key):
    # 从本地 IPFS 节点拉取文件内容
    url = f"http://127.0.0.1:8080/ipfs/{file_hash}"
    response = requests.get(url)
    response.raise_for_status()

    file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], file_hash)

    with open(file_path, 'wb') as f:
        f.write(response.content)

    decrypt_file(file_path, file_key)

    with open(file_path, 'rb') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]

    file_extension = last_line.decode()
    saved_file = file_path + '.' + file_extension
    os.rename(file_path, saved_file)

    print(saved_file)
    return saved_file


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html' , message = "Welcome!")

@app.route('/download')
def download():
    return render_template('download.html' , message = "Welcome!")

@app.route('/connect_blockchain')
def connect_blockchain():
    is_chain_replaced = blockchain.replace_chain()
    return render_template('connect_blockchain.html', chain = blockchain.chain, nodes = len(blockchain.nodes))

@app.errorhandler(413)
def entity_too_large(e):
    return render_template('upload.html' , message = "Requested Entity Too Large!")

@app.route('/add_file', methods=['POST'])
def add_file():
    
    is_chain_replaced = blockchain.replace_chain()

    if is_chain_replaced:
        print('The nodes had different chains so the chain was replaced by the longest one.')
    else:
        print('All good. The chain is the largest one.')

    if request.method == 'POST':
        error_flag = True
        if 'file' not in request.files:
            message = 'No file part'
        else:
            user_file = request.files['file']
            if user_file.filename == '':
                message = 'No file selected for uploading'

            if user_file and allowed_file(user_file.filename):
                error_flag = False
                filename = secure_filename(user_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                user_file.save(file_path)
                append_file_extension(user_file, file_path)
                sender = request.form['sender_name']
                receiver = request.form['receiver_name']
                file_key = request.form['file_key']

                try:
                    hashed_output1 = hash_user_file(file_path, file_key)
                    index = blockchain.add_file(sender, receiver, hashed_output1)
                except Exception as err:
                    message = str(err)
                    error_flag = True
                    if "ConnectionError:" in message:
                        message = "Gateway down or bad Internet!"

            else:
                error_flag = True
                message = 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'
    
        if error_flag == True:
            return render_template('upload.html' , message = message)
        else:
            return render_template('upload.html' , message = "File succesfully uploaded")

# @app.route('/retrieve_file', methods=['POST'])
# def retrieve_file():
#
#     is_chain_replaced = blockchain.replace_chain()
#
#     if is_chain_replaced:
#         print('The nodes had different chains so the chain was replaced by the longest one.')
#     else:
#         print('All good. The chain is the largest one.')
#
#     if request.method == 'POST':
#
#         error_flag = True
#
#         if request.form['file_hash'] == '':
#             message = 'No file hash entered.'
#         elif request.form['file_key'] == '':
#             message = 'No file key entered.'
#         else:
#             error_flag = False
#             file_key = request.form['file_key']
#             file_hash = request.form['file_hash']
#             try:
#                 file_path = retrieve_from_hash(file_hash, file_key)
#             except Exception as err:
#                 message = str(err)
#                 error_flag = True
#                 if "ConnectionError:" in message:
#                     message = "Gateway down or bad Internet!"
#
#         if error_flag == True:
#             return render_template('download.html' , message = message)
#         else:
#             return render_template('download.html' , message = "File successfully downloaded")

from flask import send_file

@app.route('/retrieve_file', methods=['POST'])
def retrieve_file():

    is_chain_replaced = blockchain.replace_chain()

    if is_chain_replaced:
        print('The nodes had different chains so the chain was replaced by the longest one.')
    else:
        print('All good. The chain is the largest one.')

    error_flag = True

    if request.form['file_hash'] == '':
        message = 'No file hash entered.'
    elif request.form['file_key'] == '':
        message = 'No file key entered.'
    else:
        error_flag = False
        file_key = request.form['file_key']
        file_hash = request.form['file_hash']
        try:
            file_path = retrieve_from_hash(file_hash, file_key)
        except Exception as err:
            message = str(err)
            error_flag = True
            if "ConnectionError:" in message:
                message = "Gateway down or bad Internet!"

    if error_flag:
        return render_template('download.html', message=message)
    else:
        # 文件下载，直接返回文件给浏览器
        return send_file(
            file_path,
            as_attachment=True,
            download_name=os.path.basename(file_path)
        )


# Getting the full Blockchain
@app.route('/get_chain', methods = ['GET'])
def get_chain():
    response = {'chain': blockchain.chain,
                'length': len(blockchain.chain)}
    return jsonify(response), 200

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    print(request)

@socketio.on('add_client_node')
def handle_node(client_node):
    print(client_node)
    blockchain.nodes.add(client_node['node_address'])
    emit('my_response', {'data': pickle.dumps(blockchain.nodes)}, broadcast = True)

@socketio.on('remove_client_node')
def handle_node(client_node):
    print(client_node)
    blockchain.nodes.remove(client_node['node_address'])
    emit('my_response', {'data': pickle.dumps(blockchain.nodes)}, broadcast = True)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    print(request)

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5111, allow_unsafe_werkzeug=True)
