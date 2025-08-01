#!/usr/bin/env python3
"""
Startup script for all hybrid MCP servers that use nlweb search
but return data in the same format as the original API servers.
"""

import subprocess
import sys
import os
import time
import signal
import atexit
from typing import List

# Server configurations
HYBRID_SERVERS = [
    {
        "name": "Hybrid Server A (E-Store Athletes)",
        "script": "hybrid_server_a.py",
        "port": "8060",
        "description": "Uses nlweb semantic search with server_a format"
    },
    {
        "name": "Hybrid Server B (TechTalk)",
        "script": "hybrid_server_b.py", 
        "port": "8061",
        "description": "Uses nlweb semantic search with server_b format"
    },
    {
        "name": "Hybrid Server C (CamelCases)",
        "script": "hybrid_server_c.py",
        "port": "8062", 
        "description": "Uses nlweb semantic search with server_c format"
    },
    {
        "name": "Hybrid Server D (Hardware Cafe)",
        "script": "hybrid_server_d.py",
        "port": "8063",
        "description": "Uses nlweb semantic search with server_d format"
    }
]

# Global list to track running processes
running_processes: List[subprocess.Popen] = []

def cleanup_processes():
    """Clean up all running server processes"""
    print("\n🛑 Shutting down all hybrid servers...")
    for process in running_processes:
        try:
            process.terminate()
            # Give process time to terminate gracefully
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate gracefully
            process.kill()
            process.wait()
        except Exception as e:
            print(f"Error terminating process: {e}")
    
    print("✅ All hybrid servers have been shut down.")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print(f"\n🚨 Received signal {signum}")
    cleanup_processes()
    sys.exit(0)

def start_server(server_config: dict) -> subprocess.Popen:
    """Start a single hybrid server"""
    script_path = os.path.join(os.path.dirname(__file__), server_config["script"])
    
    if not os.path.exists(script_path):
        print(f"❌ Server script not found: {script_path}")
        return None
    
    print(f"🚀 Starting {server_config['name']} on port {server_config['port']}...")
    print(f"   📄 Script: {server_config['script']}")
    print(f"   📝 Description: {server_config['description']}")
    
    try:
        # Set environment variables for the server
        env = os.environ.copy()
        env["PORT"] = server_config["port"]
        env["TRANSPORT"] = "sse"  # Use SSE transport for HTTP access
        
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ {server_config['name']} started successfully (PID: {process.pid})")
            print(f"   🌐 Available at: http://localhost:{server_config['port']}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start {server_config['name']}")
            print(f"   📤 stdout: {stdout}")
            print(f"   📥 stderr: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting {server_config['name']}: {e}")
        return None

def check_prerequisites():
    """Check if all prerequisites are available"""
    print("🔍 Checking prerequisites...")
    
    # Check if nlweb_mcp directory exists
    nlweb_path = os.path.join(os.path.dirname(__file__), '..', 'nlweb_mcp')
    if not os.path.exists(nlweb_path):
        print(f"❌ nlweb_mcp directory not found at: {nlweb_path}")
        return False
    
    # Check if required nlweb files exist
    required_files = [
        'elasticsearch_client.py',
        'embedding_service.py', 
        'search_engine.py',
        'config.py'
    ]
    
    for file in required_files:
        file_path = os.path.join(nlweb_path, file)
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return False
    
    print("✅ All prerequisites found")
    return True

def main():
    """Main function to start all hybrid servers"""
    print("🔧 WebMall Hybrid MCP Servers Startup Script")
    print("=" * 50)
    print("This script starts hybrid servers that use nlweb semantic search")
    print("but return data in the same formats as the original API servers.")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please ensure nlweb_mcp is properly set up.")
        sys.exit(1)
    
    # Register cleanup handlers
    atexit.register(cleanup_processes)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"\n🚀 Starting {len(HYBRID_SERVERS)} hybrid servers...")
    
    # Start all servers
    for server_config in HYBRID_SERVERS:
        process = start_server(server_config)
        if process:
            running_processes.append(process)
        print()  # Add spacing between server starts
    
    if not running_processes:
        print("❌ No servers started successfully. Exiting.")
        sys.exit(1)
    
    print(f"✅ Successfully started {len(running_processes)}/{len(HYBRID_SERVERS)} servers")
    print("\n📋 Server Summary:")
    print("-" * 60)
    for i, server_config in enumerate(HYBRID_SERVERS):
        if i < len(running_processes):
            status = "🟢 RUNNING"
            url = f"http://localhost:{server_config['port']}"
        else:
            status = "🔴 FAILED"
            url = "N/A"
        
        print(f"{server_config['name']:<30} {status} {url}")
    
    print("-" * 60)
    print("\n💡 Usage Notes:")
    print("• These servers use powerful nlweb semantic search capabilities")
    print("• But return data in the same heterogeneous formats as the original servers")
    print("• This enables direct comparison between the two approaches")
    print("• All servers use SSE transport and are accessible via HTTP")
    
    print("\n🛑 Press Ctrl+C to stop all servers")
    
    try:
        # Keep the script running and monitor processes
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for i, process in enumerate(running_processes[:]):
                if process.poll() is not None:
                    server_name = HYBRID_SERVERS[i]["name"]
                    print(f"\n⚠️  {server_name} has stopped unexpectedly")
                    running_processes.remove(process)
                    
                    # Optionally restart the server
                    print(f"🔄 Attempting to restart {server_name}...")
                    new_process = start_server(HYBRID_SERVERS[i])
                    if new_process:
                        running_processes.append(new_process)
                        print(f"✅ {server_name} restarted successfully")
                    else:
                        print(f"❌ Failed to restart {server_name}")
            
            # If all processes are dead, exit
            if not running_processes:
                print("\n❌ All servers have stopped. Exiting.")
                break
                
    except KeyboardInterrupt:
        print("\n🚨 Received keyboard interrupt")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        cleanup_processes()

if __name__ == "__main__":
    main()