# models.py
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), default='user')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    username = db.Column(db.String(80), nullable=False)  # 新增：直接存储用户名
    action = db.Column(db.String(50), nullable=False)  # 扩大长度
    filename = db.Column(db.String(255), nullable=True)  # 扩大长度，可为空
    details = db.Column(db.String(255), nullable=True)  # 新增：操作详情
    ip_address = db.Column(db.String(45), nullable=True)  # 新增：IP地址
    status = db.Column(db.String(20), nullable=False)  # 新增：操作状态
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)