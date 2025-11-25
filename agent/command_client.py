#!/usr/bin/env python3
"""
Command client that runs in the controller container.
Sends commands and receives responses with proper synchronization.
"""

import argparse
import json
import sys
import time
import uuid

COMMAND_PIPE = "/shared/command_pipe"
RESPONSE_PIPE = "/shared/response_pipe"


def send_command(command, working_dir=None, timeout=30):
    """Send a command and wait for the response"""
    request_id = str(uuid.uuid4())

    # Prepare request
    request = {"id": request_id, "command": command}

    # Add working directory if specified
    if working_dir:
        request["working_dir"] = working_dir

    # Send command
    with open(COMMAND_PIPE, "w") as f:
        f.write(json.dumps(request) + "\n")
        f.flush()

    # Read response with delimiter
    response_text = ""
    start_time = time.time()

    with open(RESPONSE_PIPE) as f:
        while True:
            if time.time() - start_time > timeout:
                return {
                    "error": "Client timeout waiting for response",
                    "stdout": "",
                    "stderr": "Response timeout",
                    "returncode": -1,
                }

            line = f.readline()
            if not line:
                time.sleep(0.01)
                continue

            response_text += line

            # Check for end delimiter
            if "###END###" in response_text:
                response_text = response_text.replace("###END###\n", "")
                break

    # Parse response
    try:
        response = json.loads(response_text.strip())

        # Verify this is our response
        if response.get("id") != request_id:
            return {
                "error": "Response ID mismatch",
                "stdout": "",
                "stderr": "Got response for different request",
                "returncode": -1,
            }

        return response
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse response: {e}", "stdout": response_text, "stderr": "", "returncode": -1}


def main():
    parser = argparse.ArgumentParser(description="Send commands to isolated container")
    parser.add_argument("command", nargs="?", help="Command to execute")
    parser.add_argument("-d", "--working-dir", help="Working directory for command execution")
    parser.add_argument("-c", "--command-arg", help="Alternative way to specify command")
    parser.add_argument("-t", "--timeout", type=int, default=30, help="Timeout in seconds (default: 30)")

    args = parser.parse_args()

    # Determine the command
    if args.command:
        command = args.command
    elif args.command_arg:
        command = args.command_arg
    else:
        print("Usage: command_client.py [-d WORKING_DIR] [-t TIMEOUT] <command>")
        print("   or: command_client.py -c <command> [-d WORKING_DIR] [-t TIMEOUT]")
        print("\nExamples:")
        print("  command_client.py 'ls -la'")
        print("  command_client.py -d /tmp 'pwd'")
        print("  command_client.py -c 'echo hello' -d /home")
        print("  command_client.py -t 60 'long-running-command'")
        sys.exit(1)

    # Send command and get response
    response = send_command(command, working_dir=args.working_dir, timeout=args.timeout)

    # Display results
    if response.get("stdout"):
        print(response["stdout"], end="")

    if response.get("stderr"):
        print(response["stderr"], file=sys.stderr, end="")

    if response.get("error"):
        print(f"\nError: {response['error']}", file=sys.stderr)

    # Exit with the same code as the command
    sys.exit(response.get("returncode", 0))


if __name__ == "__main__":
    main()
