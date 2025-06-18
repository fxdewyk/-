from flask import Blueprint, request, jsonify, send_file
import os
import logging
from werkzeug.utils import secure_filename
from io import BytesIO
from datetime import datetime
from models import db, AuditLog, User
from auth import verify_token
from crypto_manager import CryptoManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FileHandler')

file_bp = Blueprint('file', __name__)
UPLOAD_FOLDER = 'secure_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 初始化加密管理器
crypto_mgr = CryptoManager()

def get_client_ip():
    """获取客户端IP地址"""
    if request.headers.getlist("X-Forwarded-For"):
        return request.headers.getlist("X-Forwarded-For")[0]
    return request.remote_addr

def log_operation(user, action, filename=None, details=None, status='success'):
    try:
        log = AuditLog(
            user_id=user.id,
            username=user.username,
            action=action,
            filename=filename[:255] if filename else None,
            details=details[:255] if details else None,
            ip_address=get_client_ip(),
            status=status
        )

        db.session.add(log)
        db.session.commit()
        return True
    except Exception as e:
        logger.error(f"日志记录失败: {str(e)}")
        return False

@file_bp.route('/api/upload', methods=['POST'])
def upload_file():
    """文件上传接口"""
    token = request.headers.get('Authorization')
    user = verify_token(token)
    if not user:
        return jsonify({"success": False, "message": "未授权"}), 401

    if 'file' not in request.files:
        log_operation(user, "上传文件", None, "未选择文件", "failed")
        return jsonify({"success": False, "message": "未选择文件"}), 400

    file = request.files['file']
    if file.filename == '':
        log_operation(user, "上传文件", None, "无效文件名", "failed")
        return jsonify({"success": False, "message": "无效文件名"}), 400

    try:
        original_filename = file.filename
        safe_filename = secure_filename(original_filename)
        file_data = file.read()

        if len(file_data) == 0:
            log_operation(user, "上传文件", original_filename, "空文件", "failed")
            return jsonify({"success": False, "message": "空文件无法上传"}), 400

        encryptor = crypto_mgr.get_encryptor()
        encrypted_data = encryptor.encrypt(file_data)

        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        with open(filepath, 'wb') as f:
            f.write(encrypted_data)

        log_operation(user, "上传文件", original_filename,
                      f"大小: {len(file_data)}字节, 算法: {crypto_mgr.current_algorithm}", "success")

        return jsonify({
            "success": True,
            "message": "文件上传成功",
            "filename": original_filename,
            "algorithm": crypto_mgr.current_algorithm,
            "size": len(file_data),
            "encrypted_size": len(encrypted_data)
        })

    except Exception as e:
        logger.error(f"上传失败: {str(e)}", exc_info=True)
        log_operation(user, "上传文件", file.filename, f"错误: {str(e)}", "failed")
        return jsonify({
            "success": False,
            "message": f"文件上传失败: {str(e)}"
        }), 500

@file_bp.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """文件下载接口"""
    token = request.headers.get('Authorization')
    user = verify_token(token)
    if not user:
        return jsonify({"success": False, "message": "未授权"}), 401

    try:
        safe_filename = secure_filename(filename)
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)

        if not os.path.exists(filepath):
            log_operation(user, "下载文件", filename, "文件不存在", "failed")
            return jsonify({"success": False, "message": "文件不存在"}), 404

        with open(filepath, 'rb') as f:
            encrypted_data = f.read()

        if len(encrypted_data) == 0:
            log_operation(user, "下载文件", filename, "文件内容为空", "failed")
            return jsonify({"success": False, "message": "文件内容为空"}), 400

        encryptor = crypto_mgr.get_encryptor()
        decrypted_data = encryptor.decrypt(encrypted_data)

        if len(decrypted_data) == 0:
            log_operation(user, "下载文件", filename, "解密失败", "failed")
            return jsonify({"success": False, "message": "解密失败，返回空数据"}), 500

        log_operation(user, "下载文件", filename,
                      f"大小: {len(decrypted_data)}字节", "success")

        file_stream = BytesIO(decrypted_data)
        file_stream.seek(0)
        return send_file(
            file_stream,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )

    except Exception as e:
        logger.error(f"下载失败: {str(e)}", exc_info=True)
        log_operation(user, "下载文件", filename, f"错误: {str(e)}", "failed")
        return jsonify({
            "success": False,
            "message": f"文件下载失败: {str(e)}"
        }), 500

@file_bp.route('/api/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """文件删除接口"""
    token = request.headers.get('Authorization')
    user = verify_token(token)
    if not user:
        return jsonify({"success": False, "message": "未授权"}), 401

    try:
        safe_filename = secure_filename(filename)
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)

        if not os.path.exists(filepath):
            log_operation(user, "删除文件", filename, "文件不存在", "failed")
            return jsonify({"success": False, "message": "文件不存在"}), 404

        file_size = os.path.getsize(filepath)
        os.remove(filepath)

        log_operation(user, "删除文件", filename,
                      f"大小: {file_size}字节", "success")

        return jsonify({
            "success": True,
            "message": "文件已永久删除",
            "filename": filename
        })

    except Exception as e:
        logger.error(f"删除失败: {str(e)}", exc_info=True)
        log_operation(user, "删除文件", filename, f"错误: {str(e)}", "failed")
        return jsonify({
            "success": False,
            "message": f"文件删除失败: {str(e)}"
        }), 500

@file_bp.route('/api/files', methods=['GET'])
def list_files():
    """文件列表接口"""
    token = request.headers.get('Authorization')
    user = verify_token(token)
    if not user:
        return jsonify({"success": False, "message": "未授权"}), 401

    try:
        upload_logs = AuditLog.query.filter_by(action='上传文件', status='success').order_by(
            AuditLog.timestamp.desc()).all()

        file_map = {}
        for log in upload_logs:
            if log.filename and log.filename not in file_map:
                filepath = os.path.join(UPLOAD_FOLDER, secure_filename(log.filename))
                if os.path.exists(filepath):
                    file_map[log.filename] = {
                        "filename": log.filename,
                        "upload_time": log.timestamp,
                        "uploader": log.username,
                        "size": os.path.getsize(filepath)
                    }

        file_list = [{
            "filename": info["filename"],
            "upload_time": info["upload_time"].strftime('%Y-%m-%d %H:%M'),
            "uploader": info["uploader"],
            "size": info["size"]
        } for info in file_map.values()]

        log_operation(user, "查看文件列表", None, f"获取到 {len(file_list)} 个文件", "success")
        return jsonify({
            "success": True,
            "files": file_list
        })

    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}", exc_info=True)
        log_operation(user, "查看文件列表", None, f"错误: {str(e)}", "failed")
        return jsonify({
            "success": False,
            "message": f"获取文件列表失败: {str(e)}"
        }), 500

@file_bp.route('/api/logs', methods=['GET'])
def get_logs():
    """日志查询接口"""
    token = request.headers.get('Authorization')
    user = verify_token(token)
    if not user:
        return jsonify({"success": False, "message": "未授权"}), 401

    try:
        logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(200).all()

        log_list = []
        for log in logs:
            log_list.append({
                "username": log.username,
                "action": log.action,
                "filename": log.filename,
                "details": log.details,
                "ip_address": log.ip_address,
                "status": log.status,
                "timestamp": log.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })

        log_operation(user, "查看审计日志", None, f"获取到 {len(log_list)} 条记录", "success")
        return jsonify({
            "success": True,
            "logs": log_list
        })

    except Exception as e:
        logger.error(f"获取日志失败: {str(e)}", exc_info=True)
        log_operation(user, "查看审计日志", None, f"错误: {str(e)}", "failed")
        return jsonify({
            "success": False,
            "message": f"获取日志失败: {str(e)}"
        }), 500

@file_bp.route('/api/set_algorithm', methods=['POST'])
def set_algorithm():
    """算法切换接口"""
    token = request.headers.get('Authorization')
    user = verify_token(token)
    if not user or user.username != 'admin':
        return jsonify({"success": False, "message": "权限不足"}), 403

    try:
        data = request.get_json()
        if not data:
            log_operation(user, "设置加密算法", None, "无效请求数据", "failed")
            return jsonify({"success": False, "message": "无效请求数据"}), 400

        algorithm = data.get('algorithm', 'AES').upper()

        if algorithm not in ['AES', 'DES']:
            log_operation(user, "设置加密算法", None, f"不支持的算法: {algorithm}", "failed")
            return jsonify({"success": False, "message": "不支持的加密算法"}), 400

        old_algorithm = crypto_mgr.current_algorithm
        crypto_mgr.current_algorithm = algorithm

        log_operation(user, "设置加密算法", None,
                      f"从 {old_algorithm} 切换到 {algorithm}", "success")

        return jsonify({
            "success": True,
            "message": f"加密算法已切换为 {algorithm}"
        })

    except Exception as e:
        logger.error(f"算法切换失败: {str(e)}", exc_info=True)
        log_operation(user, "设置加密算法", None, f"错误: {str(e)}", "failed")
        return jsonify({
            "success": False,
            "message": f"算法切换失败: {str(e)}"
        }), 500