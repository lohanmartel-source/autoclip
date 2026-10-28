"""Ed25519 en Python pur (RFC 8032), juste ce qu'il faut pour signer et vérifier les codes Pro.

Lent comparé à une librairie en C, mais une vérification prend quelques millisecondes : ça suffit ici.
"""
from __future__ import annotations

import hashlib

p = 2 ** 255 - 19
L = 2 ** 252 + 27742317777372353535851937790883648493
d = (-121665 * pow(121666, p - 2, p)) % p
I = pow(2, (p - 1) // 4, p)


def _h(m: bytes) -> int:
    return int.from_bytes(hashlib.sha512(m).digest(), "little")


def _inv(x: int) -> int:
    return pow(x, p - 2, p)


def _x_depuis_y(y: int, signe: int) -> int | None:
    xx = (y * y - 1) * _inv(d * y * y + 1) % p
    x = pow(xx, (p + 3) // 8, p)
    if (x * x - xx) % p:
        x = x * I % p
    if (x * x - xx) % p:
        return None
    if x & 1 != signe:
        x = p - x
    return x


_By = 4 * _inv(5) % p
B = (_x_depuis_y(_By, 0), _By, 1, _x_depuis_y(_By, 0) * _By % p)   # coordonnées étendues


def _add(P, Q):
    x1, y1, z1, t1 = P
    x2, y2, z2, t2 = Q
    a = (y1 - x1) * (y2 - x2) % p
    b = (y1 + x1) * (y2 + x2) % p
    c = 2 * t1 * t2 * d % p
    dd = 2 * z1 * z2 % p
    e, f, g, h = b - a, dd - c, dd + c, b + a
    return (e * f % p, g * h % p, f * g % p, e * h % p)


def _mul(s: int, P):
    Q = (0, 1, 1, 0)
    while s:
        if s & 1:
            Q = _add(Q, P)
        P = _add(P, P)
        s >>= 1
    return Q


def _encoder(P) -> bytes:
    x, y, z, _ = P
    zi = _inv(z)
    x, y = x * zi % p, y * zi % p
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decoder(b: bytes):
    if len(b) != 32:
        return None
    y = int.from_bytes(b, "little")
    signe, y = y >> 255, y & ((1 << 255) - 1)
    if y >= p:
        return None
    x = _x_depuis_y(y, signe)
    if x is None:
        return None
    return (x, y, 1, x * y % p)


def _egaux(P, Q) -> bool:
    x1, y1, z1, _ = P
    x2, y2, z2, _ = Q
    return (x1 * z2 - x2 * z1) % p == 0 and (y1 * z2 - y2 * z1) % p == 0


def _secret(graine: bytes) -> tuple[int, bytes]:
    h = hashlib.sha512(graine).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def cle_publique(graine: bytes) -> bytes:
    a, _ = _secret(graine)
    return _encoder(_mul(a, B))


def signer(graine: bytes, message: bytes) -> bytes:
    a, prefixe = _secret(graine)
    A = _encoder(_mul(a, B))
    r = _h(prefixe + message) % L
    R = _encoder(_mul(r, B))
    s = (r + _h(R + A + message) * a) % L
    return R + s.to_bytes(32, "little")


def verifier(publique: bytes, message: bytes, signature: bytes) -> bool:
    if len(signature) != 64:
        return False
    A = _decoder(publique)
    R = _decoder(signature[:32])
    if A is None or R is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= L:
        return False
    k = _h(signature[:32] + publique + message) % L
    return _egaux(_mul(s, B), _add(R, _mul(k, A)))
