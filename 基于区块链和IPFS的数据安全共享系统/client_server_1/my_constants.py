# from flask import Flask
#
# UPLOAD_FOLDER = '/Users/souviksaha/Desktop/Blockchain-based-Decentralized-File-Sharing-System-using-IPFS/client_server_1/uploads'
# DOWNLOAD_FOLDER = '/Users/souviksaha/Desktop/Blockchain-based-Decentralized-File-Sharing-System-using-IPFS/client_server_1/downloads'
#
# app = Flask(__name__)
# app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
# app.config['SERVER_IP'] = '127.0.0.1:5111'
#
# app.config['NODE_ADDR'] = {'Host': '127.0.0.1', 'Port': 5113}
#
# #app.config['NODE_ADDR'] = {'Host' : '0.0.0.0', 'Port' : 5113}
# app.config['BUFFER_SIZE'] = 64 * 1024
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024




import os
from flask import Flask

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DOWNLOAD_FOLDER = os.path.join(BASE_DIR, 'downloads')

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# 修改为你的本地 server 地址
app.config['SERVER_IP'] = '127.0.0.1:5111'
app.config['NODE_ADDR'] = {'Host': '0.0.0.0', 'Port': 5113}

app.config['BUFFER_SIZE'] = 64 * 1024
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
