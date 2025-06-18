import os
from flask import Flask, render_template, redirect, request
from flask_cors import CORS
from models import db
from auth import auth_bp
from file_handler import file_bp
from spider import spider_bp
from flask_migrate import Migrate

app = Flask(__name__)
CORS(app)

# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# 注册蓝图
app.register_blueprint(auth_bp)
app.register_blueprint(file_bp)
app.register_blueprint(spider_bp)


# 首页：登录页
@app.route('/')
def index():
    return render_template('index.html')


# 登录后主页面
@app.route('/main')
def main_page():
    return render_template('main.html')


# 漏洞挖掘页面
@app.route('/mining')
def mining_page():
    return render_template('mining.html')


# 审计页面
@app.route('/audit')
def audit_page():
    return render_template('audit.html')



migrate = Migrate(app, db)

# 初始化数据库
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)  # 指定端口为5000