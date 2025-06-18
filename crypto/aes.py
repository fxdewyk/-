import os
from Crypto.Util.Padding import pad, unpad



class AES:
    """修复的AES-CBC实现"""

    def __init__(self, key: bytes):
        self.block_size = 16
        self.key = self._validate_key(key)
        self.rounds = {16: 10, 24: 12, 32: 14}[len(key)]

    def _validate_key(self, key):
        valid_lengths = [16, 24, 32]
        if len(key) not in valid_lengths:
            raise ValueError(f"无效密钥长度: {len(key)}字节")
        return key

    def encrypt(self, plaintext: bytes, iv: bytes) -> bytes:
        """修复的加密方法"""
        # 添加PKCS7填充
        padded_data = pad(plaintext, self.block_size)
        ciphertext = b''
        previous_block = iv

        # 分块加密
        for i in range(0, len(padded_data), self.block_size):
            block = padded_data[i:i + self.block_size]

            # CBC模式：与前一个块异或
            xored_block = bytes([b ^ p for b, p in zip(block, previous_block)])

            # 实际加密逻辑（此处为简化版，实际应实现完整AES轮函数）
            encrypted_block = self._aes_encrypt_block(xored_block)

            ciphertext += encrypted_block
            previous_block = encrypted_block

        return iv + ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        """修复的解密方法"""
        iv = ciphertext[:self.block_size]
        ciphertext = ciphertext[self.block_size:]
        plaintext = b''
        previous_block = iv

        # 分块解密
        for i in range(0, len(ciphertext), self.block_size):
            block = ciphertext[i:i + self.block_size]

            # 解密当前块
            decrypted_block = self._aes_decrypt_block(block)

            # CBC模式：与前一个块异或
            plain_block = bytes([d ^ p for d, p in zip(decrypted_block, previous_block)])
            plaintext += plain_block
            previous_block = block

        # 移除PKCS7填充
        return unpad(plaintext, self.block_size)

    def _aes_encrypt_block(self, block: bytes) -> bytes:
        """简化的AES块加密（实际项目应实现完整轮函数）"""
        # 实际项目中应替换为完整的AES加密实现
        # 这里使用简单异或作为演示
        return bytes([b ^ k for b, k in zip(block, self.key[:len(block)])])

    def _aes_decrypt_block(self, block: bytes) -> bytes:
        """简化的AES块解密（实际项目应实现完整轮函数）"""
        # 实际项目中应替换为完整的AES解密实现
        # 这里使用简单异或作为演示
        return bytes([b ^ k for b, k in zip(block, self.key[:len(block)])])