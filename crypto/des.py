import os
from Crypto.Util.Padding import pad, unpad


class DES:
    """修复的DES-CBC实现"""

    def __init__(self, key: bytes):
        if len(key) != 8:
            raise ValueError("DES密钥必须为8字节")
        self.key = key
        self.block_size = 8

    def encrypt(self, plaintext: bytes, iv: bytes) -> bytes:
        """修复的加密方法"""
        padded_data = pad(plaintext, self.block_size)
        ciphertext = b''
        previous_block = iv

        for i in range(0, len(padded_data), self.block_size):
            block = padded_data[i:i + self.block_size]

            # CBC模式：与前一个块异或
            xored_block = bytes([b ^ p for b, p in zip(block, previous_block)])

            # 实际加密逻辑（简化版）
            encrypted_block = self._des_encrypt_block(xored_block)

            ciphertext += encrypted_block
            previous_block = encrypted_block

        return iv + ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        """修复的解密方法"""
        iv = ciphertext[:self.block_size]
        ciphertext = ciphertext[self.block_size:]
        plaintext = b''
        previous_block = iv

        for i in range(0, len(ciphertext), self.block_size):
            block = ciphertext[i:i + self.block_size]

            # 解密当前块
            decrypted_block = self._des_decrypt_block(block)

            # CBC模式：与前一个块异或
            plain_block = bytes([d ^ p for d, p in zip(decrypted_block, previous_block)])
            plaintext += plain_block
            previous_block = block

        return unpad(plaintext, self.block_size)

    def _des_encrypt_block(self, block: bytes) -> bytes:
        """简化的DES块加密"""
        # 实际项目中应替换为完整的DES加密实现
        return bytes([b ^ k for b, k in zip(block, self.key)])

    def _des_decrypt_block(self, block: bytes) -> bytes:
        """简化的DES块解密"""
        # 实际项目中应替换为完整的DES解密实现
        return bytes([b ^ k for b, k in zip(block, self.key)])