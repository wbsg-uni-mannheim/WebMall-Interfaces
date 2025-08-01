#!/usr/bin/env python3
"""
Hybrid Server Manager class for controlling hybrid MCP servers lifecycle
Based on the NLWeb ServerManager pattern but adapted for hybrid servers
"""

import subprocess
import sys
import os
import time
import threading
from typing import List, Dict, Any, Optional

class HybridServerManager:
    """Manager class for hybrid MCP servers lifecycle"""
    
    def __init__(self, show_logs: bool = False):
        self.processes = []
        self.show_logs = show_logs
        self.log_threads = []
        
        # Hybrid server configurations
        self.server_configs = [
            {
                "name": "Hybrid Server A (E-Store Athletes)",
                "script": "hybrid_server_a.py",
                "port": "8060",
                "shop_id": "webmall_1",
                "description": "Uses nlweb semantic search with server_a format"
            },
            {
                "name": "Hybrid Server B (TechTalk)",
                "script": "hybrid_server_b.py", 
                "port": "8061",
                "shop_id": "webmall_2",
                "description": "Uses nlweb semantic search with server_b format"
            },
            {
                "name": "Hybrid Server C (CamelCases)",
                "script": "hybrid_server_c.py",
                "port": "8062",
                "shop_id": "webmall_3", 
                "description": "Uses nlweb semantic search with server_c format"
            },
            {
                "name": "Hybrid Server D (Hardware Cafe)",
                "script": "hybrid_server_d.py",
                "port": "8063",
                "shop_id": "webmall_4",
                "description": "Uses nlweb semantic search with server_d format"
            }
        ]
    
    def _stream_output(self, process, server_name: str, stream_name: str):
        """Stream output from subprocess to console with server name prefix"""
        try:
            stream = getattr(process, stream_name)
            for line in iter(stream.readline, ''):
                if line:
                    decoded_line = line.rstrip()
                    if decoded_line:
                        print(f"[{server_name.upper()}] {decoded_line}")
                        sys.stdout.flush()
        except Exception as e:
            print(f"[{server_name.upper()}] Error reading {stream_name}: {e}")
        finally:
            try:
                stream.close()
            except:
                pass
    
    def start_server(self, server_config: Dict[str, str], debug: bool = False) -> bool:
        """Start a single hybrid server"""
        script_path = os.path.join(os.path.dirname(__file__), server_config["script"])
        
        if not os.path.exists(script_path):
            print(f"✗ Server script not found: {script_path}")
            return False
        
        server_name = server_config["name"]
        print(f"Starting {server_name}...")
        
        try:
            # Set environment variables for the server
            env = os.environ.copy()
            env["PORT"] = server_config["port"]
            env["TRANSPORT"] = "sse"  # Use SSE transport for HTTP access
            
            # Configure stdout/stderr handling based on show_logs setting
            if self.show_logs:
                stdout_setting = subprocess.PIPE
                stderr_setting = subprocess.STDOUT  # Merge stderr into stdout
            else:
                stdout_setting = subprocess.PIPE
                stderr_setting = subprocess.PIPE
            
            # Start the server process
            process = subprocess.Popen(
                [sys.executable, script_path],
                env=env,
                stdout=stdout_setting,
                stderr=stderr_setting,
                cwd=os.path.dirname(__file__),
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            proc_info = {
                'server_name': server_name,
                'server_config': server_config,
                'process': process,
                'script_path': script_path
            }
            self.processes.append(proc_info)
            
            # Start log streaming threads if enabled
            if self.show_logs:
                stdout_thread = threading.Thread(
                    target=self._stream_output,
                    args=(process, server_config["shop_id"], 'stdout'),
                    daemon=True
                )
                stdout_thread.start()
                self.log_threads.append(stdout_thread)
                
                print(f"✓ {server_name} started (PID: {process.pid}) - logs will be shown below")
            else:
                print(f"✓ {server_name} started (PID: {process.pid})")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to start {server_name}: {e}")
            return False
    
    def start_all_servers(self, debug: bool = False) -> bool:
        """Start all hybrid MCP servers"""
        print("Starting all Hybrid MCP servers...")
        
        success_count = 0
        for server_config in self.server_configs:
            if self.start_server(server_config, debug):
                success_count += 1
            time.sleep(1)  # Small delay between starts
        
        print(f"\nStarted {success_count}/{len(self.server_configs)} hybrid servers successfully")
        
        if success_count > 0:
            print("\nHybrid server status:")
            for proc_info in self.processes:
                server_config = proc_info['server_config']
                pid = proc_info['process'].pid
                port = server_config['port']
                print(f"  {server_config['shop_id']}: PID {pid}, Port {port}")
        
        return success_count == len(self.server_configs)
    
    def stop_all_servers(self):
        """Stop all running hybrid servers"""
        print("\nStopping all hybrid servers...")
        
        for proc_info in self.processes:
            server_name = proc_info['server_name']
            process = proc_info['process']
            
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✓ Stopped {server_name}")
            except subprocess.TimeoutExpired:
                print(f"⚠ Force killing {server_name}")
                process.kill()
                process.wait()
            except Exception as e:
                print(f"✗ Error stopping {server_name}: {e}")
        
        # Wait for log threads to finish
        if self.show_logs:
            print("Waiting for log threads to finish...")
            for thread in self.log_threads:
                thread.join(timeout=2)
        
        # Clear the process and thread lists for next use
        self.processes.clear()
        self.log_threads.clear()
    
    def check_server_status(self) -> int:
        """Check the status of all hybrid servers"""
        print("Checking hybrid server status...")
        
        running_count = 0
        for proc_info in self.processes:
            server_config = proc_info['server_config']
            process = proc_info['process']
            
            if process.poll() is None:
                print(f"  ✓ {server_config['shop_id']}: Running (PID {process.pid})")
                running_count += 1
            else:
                print(f"  ✗ {server_config['shop_id']}: Not running (exit code: {process.returncode})")
        
        print(f"\n{running_count}/{len(self.processes)} hybrid servers running")
        return running_count
    
    def wait_for_servers(self):
        """Wait for servers and handle shutdown"""
        try:
            print("\nHybrid servers running. Press Ctrl+C to stop all servers.")
            
            while True:
                time.sleep(5)
                
                # Check if any servers have died
                running = self.check_server_status()
                if running == 0:
                    print("All hybrid servers have stopped.")
                    break
                    
        except KeyboardInterrupt:
            print("\nReceived shutdown signal...")
            self.stop_all_servers()