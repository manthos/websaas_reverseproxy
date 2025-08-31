import glob
import json
import os
import time
import uuid
import base64
import shlex
import subprocess
import logging
import mimetypes
import re
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from contextlib import asynccontextmanager

import jwt
import ffmpeg
from PIL import Image
from dotenv import load_dotenv

try:
    from starlette.datastructures import UploadFile
except ImportError:
    from fastapi import UploadFile

from fastapi import FastAPI, Request, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- OLD CONFIG LOADER ---
project_root = os.path.abspath(os.path.dirname(__file__))
#CONFIG_FILE = os.path.join(project_root, "config.json")
#with open(CONFIG_FILE, "r") as f:
#    CONFIG = json.load(f)
#def get_config():
#    return CONFIG

# --- AUTH SERVICE ---
SAAS_PUBLIC_KEY_PATH = os.getenv("SAAS_PUBLIC_KEY_PATH", os.path.join(project_root, "keys", "saas_public.pem"))
FASTAPI_PRIVATE_KEY_PATH = os.getenv("FASTAPI_PRIVATE_KEY_PATH", os.path.join(project_root, "keys", "fastapi_private.pem"))
def load_keys():
    with open(SAAS_PUBLIC_KEY_PATH, "r") as f:
        saas_public_key = f.read()
    with open(FASTAPI_PRIVATE_KEY_PATH, "r") as f:
        fastapi_private_key = f.read()
    return saas_public_key, fastapi_private_key
saas_public_key, fastapi_private_key = load_keys()
security = HTTPBearer()
class UserToken:
    def __init__(self, sub, role, exp, **kwargs):
        self.sub = sub
        self.role = role
        self.exp = exp
        self.iat = kwargs.get('iat', None)
        # Optionally store or ignore extra fields
def verify_tokens(
        authorization: HTTPAuthorizationCredentials = Depends(security),
        x_user_token: str = Header(None),
):
    server_token = authorization.credentials
    try:
        server_payload = jwt.decode(
            server_token,
            saas_public_key,
            algorithms=["RS256"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid server authentication: {str(e)}"
        )
    if not x_user_token:
        raise HTTPException(
            status_code=401,
            detail="User token missing"
        )
    try:
        user_payload = jwt.decode(
            x_user_token,
            saas_public_key,
            algorithms=["RS256"]
        )
        user = UserToken(**user_payload)
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid user authentication: {str(e)}"
        )

    # --- IAT CHECK: reject if issued more than 1 hour ago ---
    if user.iat is not None:
        now = int(time.time())
        max_age_seconds = 3600  # 1 hour
        if now - int(user.iat) > max_age_seconds:
            raise HTTPException(
                status_code=401,
                detail="Token issued too long ago"
            )

    return {"server": server_payload, "user": user}
def create_response_token(form_id):
    payload = {
        "iss": os.getenv("SERVER_ID"),
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 5,
    }
    return jwt.encode(payload, fastapi_private_key, algorithm="RS256")

# --- LIMIT CHECK SERVICE ---
class LimitCheckService:
    @staticmethod
    def model_limit_check(role, config_key, form: Dict) -> Dict:
        schema = MODEL_CONFIGS[config_key]
        role_limits = schema.get('role_limits', {}).get(role, [])
        allowed = True
        restricted_at = {}
        input_id = schema.get('main_input_id')
        input_file = form.get(input_id) if input_id else None
        for limit in role_limits:
            media_value = None
            if limit['attribute'] == 'duration':
                media_value = LimitCheckService.get_media_duration(input_file)
            elif limit['attribute'] == 'resolution':
                media_value = LimitCheckService.get_media_resolution(input_file)
            if media_value and float(media_value) > float(limit['value']):
                allowed = False
                suffix = "s" if limit['attribute'] == "duration" else "px"
                restricted_at = {
                    "title": "Exceeds the plan limit",
                    "description": f"Input exceeds the {limit['attribute']} limit {limit['value']}{suffix}, upgrade your plan"
                }
        return {
            'allowed': allowed,
            "restricted_at": restricted_at
        }
    @staticmethod
    def get_media_duration(file_path) -> int:
        probe = ffmpeg.probe(file_path)
        duration = float(probe['format']['duration'])
        return int(duration)
    @staticmethod
    def get_media_resolution(file_path) -> int:
        try:
            with Image.open(file_path) as img:
                width, _ = img.size
                return int(width or 0)
        except Exception:
            try:
                probe = ffmpeg.probe(file_path)
                for stream in probe['streams']:
                    if stream['codec_type'] == 'video':
                        return int(stream.get('width', 0))
            except Exception:
                pass
        return 0  # Ensure an int is always returned

# --- COMMAND EXECUTION SERVICE ---
class CommandExecutionService:
    def __init__(self, upload_dir: Path = Path("uploads")):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(exist_ok=True)
    def _generate_unique_filename(self, file_ext) -> Path:
        return self.upload_dir / f"{uuid.uuid4()}.{file_ext}"
    def parse_form_data(self, form_data: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        form_fields = {}
        files = {}
        logging.info(f"Received form element names: {list(form_data.keys())}")
        for key in form_data.keys():
            value = form_data[key]
            logging.info(f"Form element '{key}' type: {type(value)}")
            # Use duck typing to detect file uploads
            if hasattr(value, "filename") and hasattr(value, "read"):
                files[key] = value
            else:
                form_fields[key] = value
        return form_fields, files
    def parse_command_output_data(self, config_key):
        processed_outputs = {}
        expected_outputs = MODEL_CONFIGS[config_key]['output']
        for output in expected_outputs:
            processed_outputs[output['id']] = str(self._generate_unique_filename(output['format']))
        return processed_outputs
    def parse_output_response(self, expected_outputs):
        response = {}
        for file_id, file_path in expected_outputs.items():
            try:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                if file_path.stat().st_size == 0:
                    raise ValueError(f"Empty file: {file_path}")
                mimetype = mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
                with file_path.open("rb") as file_object:
                    file_content = file_object.read()
                    base64_encoded = base64.b64encode(file_content).decode("utf-8")
                    response[file_id] = f'data:{mimetype};base64,{base64_encoded}'
            except (IOError, OSError) as e:
                logging.error(f"Error processing file {file_path}: {e}")
                response[file_id] = None
            except Exception as e:
                logging.error(f"Unexpected error processing {file_path}: {e}")
                raise
        return response
    async def prepare_files(self, files: Dict[str, UploadFile]) -> Dict[str, Path]:
        logging.info("prepare_files called")
        uploaded_files: Dict[str, Path] = {}
        try:
            for filename, file in files.items():
                unique_filename = self._generate_unique_filename(file.filename or "")
                content = await file.read()
                with unique_filename.open("wb") as f:
                    f.write(content)
                logging.info(f"Saved uploaded file '{filename}' as '{unique_filename}'")
                uploaded_files[filename] = unique_filename
            logging.info(f"Uploaded files dict: {uploaded_files}")
            return uploaded_files
        except Exception as e:
            for uploaded_file in uploaded_files.values():
                uploaded_file.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    def sanitize_command(self, command: str) -> List[str]:
        try:
            return shlex.split(command)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid command format: {str(e)}")
    def process_command(self, command_parts: List[str], inputs: Dict[str, Any]) -> List[str]:
        processed = []
        i = 0
        while i < len(command_parts):
            part = command_parts[i]
            pattern = re.compile(r'<(.*?)>')
            match = pattern.search(part)
            if match:
                placeholder = match.group(1)
                value = inputs.get(placeholder)
                if isinstance(value, bool):
                    if value:
                        if part == f"<{placeholder}>":
                            processed.append(f"--{placeholder.replace('_', '-')}")
                        else:
                            processed.append(pattern.sub(str(value), part))
                elif value is not None and value != "":
                    #quoted_value = f'"{str(value)}"'
                    quoted_value = str(value)
                    processed.append(pattern.sub(quoted_value, part))
                else:
                    if processed and processed[-1].startswith("-"):
                        processed.pop()
            elif part.startswith("--"):
                flag_name = part.lstrip("-").replace("-", "_")
                value = inputs.get(flag_name)
                if isinstance(value, bool):
                    if value:
                        processed.append(part)
                elif value is None:
                    pass
                else:
                    processed.append(part)
            else:
                processed.append(part)
            i += 1
        return processed
    def execute_command(self, processed_command: List[str]) -> Dict[str, str | int]:
        try:
            result = subprocess.run(
                processed_command,
                capture_output=True,
                text=True,
                check=True,
                timeout=1000
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="Command execution timed out")
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Command execution failed: {e.stderr}"
            )
    async def form_to_final(self, form, config_key, role):
        parsed_form_data, files = self.parse_form_data(form)
        logging.info(f"Files dict after parsing: {files}")
        command = parsed_form_data["command"]
        if not command:
            raise HTTPException(status_code=400, detail="Command is required")
        expected_outputs = self.parse_command_output_data(config_key)
        uploaded_files = await self.prepare_files(files)
        uploaded_files_string_paths = {
            key: str(value) for key, value in uploaded_files.items()
        }
        limit_check = LimitCheckService.model_limit_check(role=role, config_key=config_key,
                                                          form=uploaded_files_string_paths)
        if not limit_check['allowed']:
            return limit_check
        input_form = json.loads(parsed_form_data['input_form'])
        command_args = {**input_form, **expected_outputs, **uploaded_files_string_paths}
        logging.info(f"Command arguments: {command_args}")
        try:
            command_parts = self.sanitize_command(command)
            processed_command = self.process_command(command_parts, command_args)
            # Log the full command line to be run
            logging.info(f"Executing command: {' '.join(processed_command)}")
            self.execute_command(processed_command)
            response = self.parse_output_response(expected_outputs)
            for _key, uploaded_file in uploaded_files.items():
                uploaded_file.unlink(missing_ok=True)
            return response
        except Exception as e:
            raise e

def load_model_configs(models_dir="models-enabled"):
    configs = {}
    for file_path in glob.glob(os.path.join(models_dir, "*.json")):
        with open(file_path, "r") as f:
            config = json.load(f)
            # Extract endpoint from actions.submit.endpoint
            endpoint_url = config.get("actions", {}).get("submit", {}).get("endpoint", "")
            if endpoint_url:
                # Get the last part of the path, e.g. 'process-image'
                endpoint = urlparse(endpoint_url).path.strip("/").split("/")[-1]
                if endpoint in configs:
                    logging.warning(f"Duplicate endpoint '{endpoint}' found in '{file_path}'. Already loaded from another file.")
                configs[endpoint] = config
    return configs

MODEL_CONFIGS = load_model_configs()

# --- FASTAPI APP ---
async def process_request(request, credentials, config_key):
    tokens = credentials
    role = tokens["user"].role
    form = await request.form()
    command_service = CommandExecutionService()
    response = await command_service.form_to_final(form, config_key, role)
    tokens = create_response_token(config_key)
    return JSONResponse(content={'outputs': response, 'token': tokens},
                        status_code=403 if not response.get('allowed', True) else 200)

def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # This code runs on startup
        logging.info(f"Listening on endpoints: {', '.join(app.state.endpoints_list)}")
        yield
        # (Optional) Add shutdown code here

    app = FastAPI(
        title="Dynamic Command Execution API",
        description="Flexible API for executing commands with file and form data support",
        version="1.0.0",
        lifespan=lifespan
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    endpoints_list = []

    for endpoint, config in MODEL_CONFIGS.items():
        route_path = f"/{endpoint}/"
        endpoints_list.append(route_path)

        def make_route(config_key):
            async def endpoint_func(
                request: Request,
                credentials: HTTPAuthorizationCredentials = Depends(verify_tokens),
                x_user_token: str = Header(None)
            ):
                return await process_request(request, credentials, config_key)
            return endpoint_func

        app.post(route_path)(make_route(endpoint))

    app.state.endpoints_list = endpoints_list  # Save for lifespan logging

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("websaas_reverseproxy:app", host="0.0.0.0", port=8000, reload=True)