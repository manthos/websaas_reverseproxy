# WebSaaS Reverse Proxy

A small FastAPI-based reverse proxy that can serve dynamically json defined AI models to an AI SaaS. It works with the AI Workflow Registry (https://github.com/manthos/aiworkflow-registry/) json files and the WebSaaS.ai no code SaaS builder.

It accepts signed requests from a SaaS provider and executes configurable command-line processing pipelines for AI generations. 

It offers two-way public-key encrypted communications between the SaaS server and an AI service node (the host that runs your AI command line).

The model definitions follow the json examples provided. The definitions also contain role and other limit information to be respected in the AI service node also (not just the SaaS side).

This repository contains: the proxy code, sample model configuration files, and utility scripts for generating keys.


## What this repo contains
- `websaas_reverseproxy.py` - main FastAPI application
- `generate_keys.py` - helper to create RSA keys (calls openssl)
- `models-enabled/` - json AI model descriptors that expose endpoints and define AI input/output and permissions
- `keys/*.example` - placeholders showing required key files (do not commit secrets)
- `uploads/` - runtime storage for output files (files should be ignored in git)

## Quickstart (local)
1. Create a Python virtual environment and install deps:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Generate keys (or copy your existing keys) and place them in `keys/`:

```bash
python generate_keys.py
# or use openssl directly if you prefer
```
Update SAAS_PUBLIC_KEY_PATH and FASTAPI_PRIVATE_KEY_PATH constants in the script and ensure they are set correctly

Review .env file and ensure keys and SERVER_ID are set 
SERVER_ID must be a stable, unique identifier agreed with the SaaS side (or stored in the model config server_id entries). It can be any string, but it must match what the SaaS expects in the iss claim. If you need to generate one locally (and will register it or tell the provider), make a random base64 id:
python - <<'PY'
import os,base64
print("base64:" + base64.b64encode(os.urandom(24)).decode())
PY

Ensure you inform the SaaS provider of the generated SERVER_ID by including it also in your model json config files (which also reside on the SaaS side)

3. Define the AI models/workflows input/outputs, endpoints and roles/limits per the json examples found in models-enabled/ (one json per model/endpoint)

4. Run/Test locally:

```bash
php websaas_reverseproxy.py
```
The parameters for the unicorn server in the script are
```
host="0.0.0.0", port=8000, reload=True
```

Then connect from your SaaS server (get a complete SaaS server from WebSaaS.ai) to test.

5. Configure so it runs automatically on each node start/boot
Add this to your crontab to automatically start the proxy on your AI service node
@reboot cd /home/user/websaas_reverseproxy/; source venv/bin/activate; (nohup python websaas_reverseproxy.py &) ;



## Connect to your SaaS or to your WebSaaS.ai SaaS (alpha version when this README is written). 
Get a WebSaaS.ai account and create a Project. Your SaaS server will be ready in minutes.
Use this repo to configure your AI Service node.
Once done, you should have your full SaaS web site and getting your own subscribers for your connected AI models!

## Read more at https://websaas.ai
