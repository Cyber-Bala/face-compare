"""Start the portal over HTTPS (phones need HTTPS for camera). Self-signed cert is generated once."""
import asyncio
import datetime
import ipaddress
import socket
import sys
from pathlib import Path

import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERT_DIR = Path(__file__).parent / "data"
CERT, KEY = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def make_cert(ip):
    CERT_DIR.mkdir(exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "face-portal")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(
            [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
             x509.IPAddress(ipaddress.ip_address(ip))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                      serialization.PrivateFormat.TraditionalOpenSSL,
                                      serialization.NoEncryption()))
    CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


async def serve_both(ip):
    """Phone page (HTTPS, LAN) and gym-backend service (HTTP, this machine only) in one process."""
    from app.main import app
    servers = [
        uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8443, ssl_certfile=str(CERT), ssl_keyfile=str(KEY))),
        uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8001)),
    ]
    print(f"Phone URL: https://{ip}:8443  (accept the self-signed cert warning)")
    print("Gym backend service: http://127.0.0.1:8001")
    await asyncio.gather(*(sv.serve() for sv in servers))


if __name__ == "__main__":
    if "--http" in sys.argv:
        # Service only (no phone page).
        print("Service mode: http://127.0.0.1:8001")
        uvicorn.run("app.main:app", host="127.0.0.1", port=8001)
        sys.exit(0)
    ip = lan_ip()
    if not CERT.exists():
        make_cert(ip)
    asyncio.run(serve_both(ip))
