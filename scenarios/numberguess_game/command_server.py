#!/usr/bin/env python3
"""
Command server that runs in the isolated container.
Executes commands and returns results with proper delimiters.
"""

import json
import os
import re
import subprocess
import time
import uuid

COMMAND_PIPE = "/shared/command_pipe"
RESPONSE_PIPE = "/shared/response_pipe"

# Store the last used working directory
last_working_dir = None


def ensure_pipes():
    """Create named pipes if they don't exist"""
    for pipe in [COMMAND_PIPE, RESPONSE_PIPE]:
        if not os.path.exists(pipe):
            os.mkfifo(pipe)


def execute_command(command, working_dir=None):
    """Execute a command and return the result"""
    global last_working_dir

    # Use specified working directory, or fall back to last used, or current directory
    cwd = working_dir or last_working_dir or os.getcwd()

    # Verify the directory exists
    if not os.path.exists(cwd):
        return {
            "stdout": "",
            "stderr": f"Working directory does not exist: {cwd}",
            "returncode": -1,
            "error": "invalid_working_dir",
            "working_dir": cwd,
        }

    # Update last working directory
    last_working_dir = cwd

    try:
        # If the command attempts to write to /output/answer.txt via shell redirection,
        # proactively replace a symlink with a regular file path so writes succeed.
        try:
            write_redir_pattern = re.compile(
                r">\s*/output/answer\.txt|>>\s*/output/answer\.txt|>\s*\./output/answer\.txt|>>\s*\./output/answer\.txt"
            )
            if write_redir_pattern.search(command):
                ans_path = "/output/answer.txt"
                if os.path.islink(ans_path):
                    os.unlink(ans_path)
        except Exception:
            # best-effort; continue even if we cannot pre-clean
            pass

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd=cwd,  # Set working directory
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": None,
            "working_dir": cwd,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Command timed out after 30 seconds",
            "returncode": -1,
            "error": "timeout",
            "working_dir": cwd,
        }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "error": "exception", "working_dir": cwd}


def main():
    ensure_pipes()
    print("Command server started", flush=True)

    while True:
        try:
            # Read command from pipe
            with open(COMMAND_PIPE) as f:
                request = f.readline().strip()

            if not request:
                continue

            # Parse the request
            try:
                req_data = json.loads(request)
                request_id = req_data.get("id", str(uuid.uuid4()))
                command = req_data.get("command", "")
                working_dir = req_data.get("working_dir", None)
            except json.JSONDecodeError:
                # Backward compatibility: treat as raw command
                request_id = str(uuid.uuid4())
                command = request
                working_dir = None

            print(f"Executing command: {command}", flush=True)
            if working_dir:
                print(f"In directory: {working_dir}", flush=True)

            # Execute the command
            result = execute_command(command, working_dir)

            # Prepare response
            response = {
                "id": request_id,
                "command": command,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"],
                "error": result["error"],
                "working_dir": result["working_dir"],
                "timestamp": time.time(),
            }

            # Write response with delimiter
            with open(RESPONSE_PIPE, "w") as f:
                f.write(json.dumps(response) + "\n###END###\n")
                f.flush()

        except KeyboardInterrupt:
            print("Server shutting down", flush=True)
            break
        except Exception as e:
            print(f"Server error: {e}", flush=True)
            # Try to send error response
            try:
                error_response = {
                    "id": "error",
                    "command": "unknown",
                    "stdout": "",
                    "stderr": f"Server error: {str(e)}",
                    "returncode": -1,
                    "error": "server_error",
                    "timestamp": time.time(),
                }
                with open(RESPONSE_PIPE, "w") as f:
                    f.write(json.dumps(error_response) + "\n###END###\n")
                    f.flush()
            except Exception:
                pass


if __name__ == "__main__":
    main()
