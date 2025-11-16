"""FingerChain 实验用的轻量级 Paillier 同态加密工具。"""
from __future__ import annotations

import math
import random
import secrets
from dataclasses import dataclass
from typing import List, Sequence, Tuple


def _l_function(x: int, n: int) -> int:
    """Paillier 解密中的 L 函数，返回 floor((x-1)/n)。"""
    return (x - 1) // n


def _lcm(a: int, b: int) -> int:
    """求最小公倍数，用于计算 λ=lcm(p-1, q-1)。"""
    return abs(a * b) // math.gcd(a, b)


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    """扩展欧几里得，用于求逆，迭代实现避免深递归。"""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b != 0:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def _modinv(a: int, n: int) -> int:
    """模逆计算，若不存在则抛异常。"""
    g, x, _ = _egcd(a, n)
    if g != 1:
        raise ValueError("Inverse does not exist")
    return x % n


def _is_probable_prime(candidate: int, rounds: int = 10) -> bool:
    """米勒拉宾测试，足够用于实验场景。"""
    if candidate < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    if candidate in small_primes:
        return True
    for prime in small_primes:
        if candidate % prime == 0:
            return False
    # write candidate - 1 as d * 2^s
    d = candidate - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = random.randrange(2, candidate - 1)
        x = pow(a, d, candidate)
        if x == 1 or x == candidate - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, candidate)
            if x == candidate - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int) -> int:
    assert bits >= 16, "Prime size too small"
    while True:
        # 生成奇数候选并设置最高位，确保指定位数
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(candidate):
            return candidate


@dataclass
class PaillierPublicKey:
    """公钥容器，负责加密与同态运算。"""

    n: int
    g: int

    @property
    def n_sq(self) -> int:
        return self.n * self.n

    def _random_r(self) -> int:
        while True:
            r = secrets.randbelow(self.n)
            if r > 0 and math.gcd(r, self.n) == 1:
                return r

    def encrypt_int(self, value: int) -> int:
        return self.encrypt(value)

    def encrypt(self, value: int, r: int | None = None) -> int:
        """对整数执行 Paillier 加密，支持自定义随机 r。"""
        value = value % self.n
        if r is None:
            r = self._random_r()
        return (pow(self.g, value, self.n_sq) * pow(r, self.n, self.n_sq)) % self.n_sq

    def homomorphic_add(self, c1: int, c2: int) -> int:
        return (c1 * c2) % self.n_sq

    def homomorphic_mul_const(self, ciphertext: int, constant: int) -> int:
        """实现 c^{k}，用于乘常数。支持负常数表示取逆。"""
        if constant == 0:
            return 1  # encryption of 0 when using g = n + 1
        if constant < 0:
            inverse = _modinv(ciphertext, self.n_sq)
            return pow(inverse, -constant, self.n_sq)
        return pow(ciphertext, constant, self.n_sq)


@dataclass
class PaillierPrivateKey:
    """私钥容器，负责解密。"""

    lambda_param: int
    mu: int
    n: int

    @property
    def n_sq(self) -> int:
        return self.n * self.n

    def decrypt(self, ciphertext: int) -> int:
        """解密并映射回 [-n/2, n/2]，方便表示负数。"""
        u = pow(ciphertext, self.lambda_param, self.n_sq)
        l = _l_function(u, self.n)
        result = (l * self.mu) % self.n
        # Convert to signed range
        if result > self.n // 2:
            result -= self.n
        return result


def generate_keypair(bits: int = 1024) -> Tuple[PaillierPublicKey, PaillierPrivateKey]:
    """生成 Paillier 公私钥对，bits 越大安全性越高但开销也越大。"""
    half = bits // 2
    p = _generate_prime(half)
    q = _generate_prime(half)
    while p == q:
        q = _generate_prime(half)
    n = p * q
    g = n + 1
    lambda_param = _lcm(p - 1, q - 1)
    n_sq = n * n
    x = pow(g, lambda_param, n_sq)
    l_val = _l_function(x, n)
    mu = _modinv(l_val, n)
    pub = PaillierPublicKey(n=n, g=g)
    priv = PaillierPrivateKey(lambda_param=lambda_param, mu=mu, n=n)
    return pub, priv


def encrypt_vector(pub: PaillierPublicKey, values: Sequence[int]) -> List[int]:
    """批量加密一组整数。"""
    return [pub.encrypt_int(v) for v in values]


def decrypt_vector(priv: PaillierPrivateKey, ciphertexts: Sequence[int]) -> List[int]:
    """批量解密一组密文。"""
    return [priv.decrypt(c) for c in ciphertexts]
