import os
from crypto.aes import AES
from crypto.des import DES


class CryptoManager:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.init_keys()
            cls._instance.current_algorithm = 'AES'
        return cls._instance

    def init_keys(self):
        self.key_dir = 'crypto_keys'
        os.makedirs(self.key_dir, exist_ok=True)

        # 修复：使用固定密钥测试（实际项目应使用随机密钥）
        self.aes_key_path = os.path.join(self.key_dir, 'aes.key')
        if not os.path.exists(self.aes_key_path):
            with open(self.aes_key_path, 'wb') as f:
                # 使用固定密钥确保一致性
                f.write(b'ThisIsA32ByteKeyForAES256Test!!')

        self.des_key_path = os.path.join(self.key_dir, 'des.key')
        if not os.path.exists(self.des_key_path):
            with open(self.des_key_path, 'wb') as f:
                # 使用固定密钥确保一致性
                f.write(b'8ByteKey')

    def get_encryptor(self, algorithm=None):
        algorithm = algorithm or self.current_algorithm
        if algorithm == 'AES':
            return self.AESEncryptor()
        elif algorithm == 'DES':
            return self.DESEncryptor()
        raise ValueError("不支持的算法")

    class AESEncryptor:
        def __init__(self):
            with open(CryptoManager().aes_key_path, 'rb') as f:
                self.key = f.read()
            self.cipher = AES(self.key)

        def encrypt(self, data):
            iv = os.urandom(16)
            return self.cipher.encrypt(data, iv)

        def decrypt(self, data):
            return self.cipher.decrypt(data)

    class DESEncryptor:
        def __init__(self):
            with open(CryptoManager().des_key_path, 'rb') as f:
                self.key = f.read()
            self.cipher = DES(self.key)

        def encrypt(self, data):
            iv = os.urandom(8)
            return self.cipher.encrypt(data, iv)

        def decrypt(self, data):
            return self.cipher.decrypt(data)