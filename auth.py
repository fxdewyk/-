from flask import Blueprint, request, jsonify, g
from models import db, User
import jwt
import datetime
import os
from functools import wraps

# 创建蓝图
auth_bp = Blueprint('auth', __name__)


# 从文件读取密钥
def get_secret_key():
    # 直接返回固定密钥，确保生成和验证token时使用相同的密钥
    return 'ping32_default_secret_key'  # 使用一个固定值


# 生成JWT Token
def generate_token(user_id):
    # 使用当前时间戳而不是datetime对象
    current_time = int(datetime.datetime.utcnow().timestamp())
    expiration_time = current_time + 24 * 60 * 60  # 24小时后过期

    payload = {
        'user_id': user_id,
        'exp': expiration_time,
        'iat': current_time
    }
    return jwt.encode(payload, get_secret_key(), algorithm='HS256')


# 验证Token
def verify_token(token):
    try:
        # 添加调试信息
        print(f"Verifying token: {token}")

        payload = jwt.decode(token, get_secret_key(), algorithms=['HS256'])
        print(f"Decoded payload: {payload}")

        user_id = payload.get('user_id')
        if not user_id:
            print("No user_id in payload")
            return None

        user = User.query.filter_by(id=user_id).first()
        print(f"Found user: {user}")

        return user
    except jwt.ExpiredSignatureError:
        print("Token expired")
        return None  # Token已过期
    except jwt.InvalidTokenError as e:
        print(f"Invalid token: {str(e)}")
        return None  # 无效Token
    except Exception as e:
        print(f"Other error: {str(e)}")
        return None  # 其他错误


# Token验证中间件
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')

        # 添加调试信息
        print(f"Auth header: {auth_header}")

        if auth_header:
            try:
                token = auth_header.split(" ")[1]  # Bearer token格式
                print(f"Extracted token: {token}")
            except IndexError:
                print("Invalid auth header format")
                return jsonify({'success': False, 'message': '无效的认证头格式'}), 401

        if not token:
            # 检查是否在cookie中
            token = request.cookies.get('token')
            if not token:
                print("No token found")
                return jsonify({'success': False, 'message': '缺少认证Token'}), 401

        user = verify_token(token)
        if not user:
            print("Invalid token or user not found")
            return jsonify({'success': False, 'message': 'Token无效或已过期'}), 401

        # 将当前用户存储在g对象中，以便视图函数访问
        g.user = user
        return f(*args, **kwargs)

    return decorated


# 注册接口
@auth_bp.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'success': False, 'message': '请提供用户名和密码'}), 400

    # 检查用户名是否已存在
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'success': False, 'message': '用户名已存在'}), 400

    # 创建新用户
    new_user = User(username=data['username'])
    new_user.set_password(data['password'])

    # 保存到数据库
    db.session.add(new_user)
    db.session.commit()

    # 生成Token
    token = generate_token(new_user.id)

    return jsonify({
        'success': True,
        'message': '注册成功',
        'token': token
    })


# 登录接口
@auth_bp.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'success': False, 'message': '请提供用户名和密码'}), 400

    # 查找用户
    user = User.query.filter_by(username=data['username']).first()

    # 验证用户和密码
    if user and user.check_password(data['password']):
        # 生成Token
        token = generate_token(user.id)

        return jsonify({
            'success': True,
            'message': '登录成功',
            'token': token
        })

    return jsonify({'success': False, 'message': '用户名或密码错误'}), 401


# 获取当前用户信息
@auth_bp.route('/api/user', methods=['GET'])
@token_required
def get_user():
    return jsonify({
        'success': True,
        'user': {
            'id': g.user.id,
            'username': g.user.username
        }
    })


# 验证Token接口（用于前端验证Token有效性）
@auth_bp.route('/api/verify_token', methods=['POST'])
def token_verify():
    data = request.get_json()

    if not data or not data.get('token'):
        return jsonify({'success': False, 'message': '请提供Token'}), 400

    user = verify_token(data['token'])

    if user:
        return jsonify({
            'success': True,
            'message': 'Token有效',
            'user': {
                'id': user.id,
                'username': user.username
            }
        })

    return jsonify({'success': False, 'message': 'Token无效或已过期'}), 401

