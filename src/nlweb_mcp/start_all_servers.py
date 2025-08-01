#!/usr/bin/env python3
"""
Start all MCP servers for the NLWeb implementation
"""

import subprocess
import sys
import os
import time
import signal
import argparse
import threading
import select

# Handle both relative and absolute imports
try:
    from .config import WEBMALL_SHOPS
except ImportError:
    from config import WEBMALL_SHOPS

class ServerManager:
    def __init__(self, show_logs: bool = False):
        self.processes = []
        self.show_logs = show_logs
        self.log_threads = []
        self.server_scripts = {
            "webmall_1": "mcp_servers/webmall_1_server.py",
            "webmall_2": "mcp_servers/webmall_2_server.py", 
            "webmall_3": "mcp_servers/webmall_3_server.py",
            "webmall_4": "mcp_servers/webmall_4_server.py"
        }
    
    def _stream_output(self, process, shop_id, stream_name):
        """Stream output from subprocess to console with shop ID prefix"""
        try:
            stream = getattr(process, stream_name)
            for line in iter(stream.readline, ''):  # Empty string for text mode
                if line:
                    # Line is already a string because we used universal_newlines=True
                    decoded_line = line.rstrip()
                    if decoded_line:  # Only print non-empty lines
                        print(f"[{shop_id.upper()}] {decoded_line}")
                        sys.stdout.flush()
        except Exception as e:
            print(f"[{shop_id.upper()}] Error reading {stream_name}: {e}")
        finally:
            try:
                stream.close()
            except:
                pass
    
    def start_server(self, shop_id: str, debug: bool = False):
        """Start a single MCP server"""
        script_path = self.server_scripts[shop_id]
        cmd = [sys.executable, script_path, '--transport', 'sse']
        
        if debug:
            cmd.append('--debug')
        
        print(f"Starting {shop_id} server...")
        
        try:
            # Configure stdout/stderr handling based on show_logs setting
            if self.show_logs:
                # Keep pipes for log streaming
                stdout_setting = subprocess.PIPE
                stderr_setting = subprocess.STDOUT  # Merge stderr into stdout
            else:
                # Keep original behavior for no-logs mode
                stdout_setting = subprocess.PIPE
                stderr_setting = subprocess.PIPE
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=stdout_setting,
                stderr=stderr_setting,
                cwd=os.path.dirname(__file__),
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            proc_info = {
                'shop_id': shop_id,
                'process': process,
                'cmd': ' '.join(cmd)
            }
            self.processes.append(proc_info)
            
            # Start log streaming threads if enabled
            if self.show_logs:
                stdout_thread = threading.Thread(
                    target=self._stream_output,
                    args=(process, shop_id, 'stdout'),
                    daemon=True
                )
                stdout_thread.start()
                self.log_threads.append(stdout_thread)
                
                print(f"✓ {shop_id} server started (PID: {process.pid}) - logs will be shown below")
            else:
                print(f"✓ {shop_id} server started (PID: {process.pid})")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to start {shop_id} server: {e}")
            return False
    
    def start_all_servers(self, debug: bool = False):
        """Start all MCP servers"""
        print("Starting all NLWeb MCP servers...")
        
        success_count = 0
        for shop_id in WEBMALL_SHOPS.keys():
            if self.start_server(shop_id, debug):
                success_count += 1
            time.sleep(1)  # Small delay between starts
        
        print(f"\nStarted {success_count}/{len(WEBMALL_SHOPS)} servers successfully")
        
        if success_count > 0:
            print("\nServer status:")
            for proc_info in self.processes:
                shop_id = proc_info['shop_id']
                pid = proc_info['process'].pid
                port = WEBMALL_SHOPS[shop_id]['mcp_port']
                print(f"  {shop_id}: PID {pid}, Port {port}")
        
        return success_count == len(WEBMALL_SHOPS)
    
    def stop_all_servers(self):
        """Stop all running servers"""
        print("\nStopping all servers...")
        
        for proc_info in self.processes:
            shop_id = proc_info['shop_id']
            process = proc_info['process']
            
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✓ Stopped {shop_id} server")
            except subprocess.TimeoutExpired:
                print(f"⚠ Force killing {shop_id} server")
                process.kill()
                process.wait()
            except Exception as e:
                print(f"✗ Error stopping {shop_id} server: {e}")
        
        # Wait for log threads to finish
        if self.show_logs:
            print("Waiting for log threads to finish...")
            for thread in self.log_threads:
                thread.join(timeout=2)
        
        # Clear the process and thread lists for next use
        self.processes.clear()
        self.log_threads.clear()
    
    def check_server_status(self):
        """Check the status of all servers"""
        print("Checking server status...")
        
        running_count = 0
        for proc_info in self.processes:
            shop_id = proc_info['shop_id']
            process = proc_info['process']
            
            if process.poll() is None:
                print(f"  ✓ {shop_id}: Running (PID {process.pid})")
                running_count += 1
            else:
                print(f"  ✗ {shop_id}: Not running (exit code: {process.returncode})")
        
        print(f"\n{running_count}/{len(self.processes)} servers running")
        return running_count
    
    def wait_for_servers(self):
        """Wait for servers and handle shutdown"""
        try:
            print("\nServers running. Press Ctrl+C to stop all servers.")
            
            while True:
                time.sleep(5)
                
                # Check if any servers have died
                running = self.check_server_status()
                if running == 0:
                    print("All servers have stopped.")
                    break
                
        except KeyboardInterrupt:
            print("\nReceived shutdown signal...")
            self.stop_all_servers()

def main():
    parser = argparse.ArgumentParser(description='Start NLWeb MCP servers')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for servers')
    parser.add_argument('--check-only', action='store_true', help='Only check server status, don\'t start')
    parser.add_argument('--show-logs', action='store_true', help='Show server logs in real-time (recommended with --debug)')
    
    args = parser.parse_args()
    
    # Show logs automatically when debug is enabled, or when explicitly requested
    show_logs = args.show_logs or args.debug
    manager = ServerManager(show_logs=show_logs)
    
    if show_logs and args.debug:
        print("Debug logging enabled - server logs will be displayed below with [SHOP_ID] prefixes")
        print("Log files are also saved to /tmp/nlweb_mcp_*.log")
        print("-" * 60)
    
    if args.check_only:
        manager.check_server_status()
        return
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}")
        manager.stop_all_servers()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start all servers
    if manager.start_all_servers(args.debug):
        manager.wait_for_servers()
    else:
        print("Failed to start all servers")
        manager.stop_all_servers()
        sys.exit(1)

if __name__ == "__main__":
    main()