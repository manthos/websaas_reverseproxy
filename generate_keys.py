import subprocess
import os

private_key_path = "keys/fastapi_private.pem"
public_key_path = "keys/fastapi_public.pem"

keys_directory = os.path.dirname(private_key_path)

if not os.path.exists(keys_directory):
    os.makedirs(keys_directory)

if not os.access(".", os.W_OK):
    print("Error: Current directory is not writable.")
    exit(1)

private_key_command = f"openssl genpkey -algorithm RSA -out {private_key_path} -pkeyopt rsa_keygen_bits:2048"
private_key_result = subprocess.run(private_key_command, shell=True, capture_output=True)

if private_key_result.returncode != 0:
    print(f"Failed to generate private key: {private_key_result.stderr.decode()}")
    exit(1)

print(f"Private key generated successfully at {private_key_path}")

public_key_command = f"openssl rsa -pubout -in {private_key_path} -out {public_key_path}"
public_key_result = subprocess.run(public_key_command, shell=True, capture_output=True)

if public_key_result.returncode != 0:
    print(f"Failed to generate public key: {public_key_result.stderr.decode()}")
    exit(1)

print(f"Public key generated successfully at {public_key_path}")
