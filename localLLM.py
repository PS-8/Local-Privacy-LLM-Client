#!/usr/bin/env python3
"""
Local Ollama chat with Model Downloader and improved UX.
- Auto-installs dependencies.
- Background diagnostics (auto-starts ollama serve).
- Real-time download progress.
- Streaming chat responses.
"""

import subprocess
import sys
import threading
import time
import shutil
import queue
import os
import importlib

# -------------------------
# Auto-install Python deps
# -------------------------
def ensure(pkg_name):
    try:
        return importlib.import_module(pkg_name)
    except Exception:
        print(f"Installing {pkg_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        return importlib.import_module(pkg_name)

ollama = ensure("ollama")

# -------------------------
# GUI imports
# -------------------------
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# -------------------------
# Configuration
# -------------------------
POPULAR_MODELS = [
    "llama3.2:1b",
    "llama3.2:3b",
    "phi3:mini",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "gemma:2b",
    "tinyllama:1.1b",
]

OLLAMA_CLI = shutil.which("ollama")
DIAG_POLL_INTERVAL = 1.0
DIAG_TIMEOUT = 30

# Thread-safe queue for UI updates
_gui_q = queue.Queue()

# -------------------------
# Utility functions
# -------------------------
def is_ollama_reachable():
    try:
        ollama.list()
        return True
    except Exception:
        return False

def get_installed_models():
    """Safely parse the installed models from the ollama client."""
    try:
        resp = ollama.list()
        models = []
        
        # Handle different versions of the ollama python client
        if hasattr(resp, 'models'):
            items = resp.models
        elif isinstance(resp, dict) and 'models' in resp:
            items = resp['models']
        else:
            items = []

        for m in items:
            if hasattr(m, 'model'):
                models.append(m.model)
            elif isinstance(m, dict) and 'model' in m:
                models.append(m['model'])
            elif isinstance(m, dict) and 'name' in m:
                models.append(m['name'])
        return sorted(list(set(models)))
    except Exception:
        return []

def start_ollama_serve_background():
    if not OLLAMA_CLI:
        _gui_q.put({"type": "status", "msg": "❌ 'ollama' CLI not found. Install from ollama.com"})
        return False

    _gui_q.put({"type": "status", "msg": "Attempting to start 'ollama serve'..."})
    try:
        if os.name == "nt":
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen(
                [OLLAMA_CLI, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=CREATE_NEW_PROCESS_GROUP, shell=False
            )
        else:
            subprocess.Popen(
                [OLLAMA_CLI, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, shell=False
            )
        return True
    except Exception as e:
        _gui_q.put({"type": "status", "msg": f"❌ Failed to start server: {e}"})
        return False

# -------------------------
# Background Threads
# -------------------------
def diagnostics_thread():
    if is_ollama_reachable():
        _gui_q.put({"type": "status", "msg": "✅ Ollama running."})
        _gui_q.put({"type": "refresh_models"})
        return

    if start_ollama_serve_background():
        start_time = time.time()
        while time.time() - start_time < DIAG_TIMEOUT:
            if is_ollama_reachable():
                _gui_q.put({"type": "status", "msg": "✅ Ollama started and reachable."})
                _gui_q.put({"type": "refresh_models"})
                return
            time.sleep(DIAG_POLL_INTERVAL)
        
        _gui_q.put({"type": "status", "msg": "❌ Timeout waiting for Ollama."})

def download_model_thread(model_name):
    try:
        _gui_q.put({"type": "status", "msg": f"⬇ Starting download: {model_name}"})
        for progress in ollama.pull(model_name, stream=True):
            status_text = progress.get('status', 'Pulling...')
            
            # Use 'or 0' to ensure we get an integer even if the API returns None
            completed = progress.get('completed') or 0
            total = progress.get('total') or 0
            
            if total > 0:
                percent = int((completed / total) * 100)
                _gui_q.put({"type": "download_progress", "percent": percent, "status": status_text})
            else:
                _gui_q.put({"type": "download_status", "status": status_text})
                
        _gui_q.put({"type": "download_done", "model": model_name})
    except Exception as e:
        _gui_q.put({"type": "download_error", "error": str(e)})

def chat_thread(model, prompt):
    try:
        if not is_ollama_reachable():
            _gui_q.put({"type": "chat_error", "error": "Ollama server is not reachable."})
            return

        stream = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=True)
        for chunk in stream:
            content = chunk['message']['content']
            _gui_q.put({"type": "chat_stream", "content": content})
            
        _gui_q.put({"type": "chat_done"})
    except Exception as e:
        _gui_q.put({"type": "chat_error", "error": str(e)})

# -------------------------
# GUI application
# -------------------------
class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Ollama Client")
        self.geometry("950x650")
        
        # Set a cleaner theme
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.setup_ui()
        
        # Start queue polling
        self.after(100, self.poll_queue)
        
        # Start background diagnostics
        threading.Thread(target=diagnostics_thread, daemon=True).start()

    def setup_ui(self):
        # Main layout: PanedWindow for Sidebar and Main Content
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Sidebar ---
        sidebar = ttk.Frame(paned, width=280, relief="flat")
        paned.add(sidebar, weight=0)

        # 1. Chat Model Selection
        ttk.Label(sidebar, text="Active Chat Model", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(5, 2))
        
        self.active_model_var = tk.StringVar()
        self.active_model_combo = ttk.Combobox(sidebar, textvariable=self.active_model_var, state="readonly")
        self.active_model_combo.pack(fill="x", pady=2)
        
        ttk.Button(sidebar, text="🔄 Refresh Installed Models", command=self.refresh_installed_models).pack(fill="x", pady=(2, 15))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", pady=10)

        # 2. Download Models
        ttk.Label(sidebar, text="Download New Model", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(5, 2))
        
        self.dl_model_var = tk.StringVar(value=POPULAR_MODELS[0])
        self.dl_model_combo = ttk.Combobox(sidebar, textvariable=self.dl_model_var, values=POPULAR_MODELS)
        self.dl_model_combo.pack(fill="x", pady=2)
        
        self.dl_btn = ttk.Button(sidebar, text="⬇ Download", command=self.start_download)
        self.dl_btn.pack(fill="x", pady=5)

        # Download Progress
        self.dl_status_var = tk.StringVar(value="")
        self.dl_status_label = ttk.Label(sidebar, textvariable=self.dl_status_var, font=("Helvetica", 8))
        self.dl_status_label.pack(anchor="w")

        self.dl_progress = ttk.Progressbar(sidebar, orient="horizontal", mode="determinate", maximum=100)
        self.dl_progress.pack(fill="x", pady=2)

        # --- Main Chat Area ---
        main_area = ttk.Frame(paned)
        paned.add(main_area, weight=1)

        self.chat_box = scrolledtext.ScrolledText(main_area, wrap="word", state="disabled", font=("Helvetica", 11))
        self.chat_box.pack(fill="both", expand=True, padx=(5, 0), pady=(0, 5))

        # Input Area
        input_frame = ttk.Frame(main_area)
        input_frame.pack(fill="x", padx=(5, 0))

        self.input_text = tk.Text(input_frame, height=4, wrap="word", font=("Helvetica", 11))
        self.input_text.pack(side="left", fill="both", expand=True)
        self.input_text.bind("<Return>", self.on_enter_key)
        self.input_text.bind("<Shift-Return>", self.on_shift_enter)

        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_btn.pack(side="left", padx=(5, 0), fill="y")

        # --- Bottom Status Bar ---
        self.status_var = tk.StringVar(value="Status: Initializing...")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", padding=2)
        status_bar.pack(side="bottom", fill="x")

        self._append_chat("System", "Welcome! If Ollama is running, models will populate automatically. You can also download new models from the sidebar.\n")

    # --- UI Logic ---
    def refresh_installed_models(self):
        models = get_installed_models()
        self.active_model_combo['values'] = models
        if models:
            if not self.active_model_var.get() or self.active_model_var.get() not in models:
                self.active_model_combo.current(0)
            self.status_var.set(f"✅ Found {len(models)} installed models.")
        else:
            self.active_model_var.set("")
            self.status_var.set("⚠ No models installed. Please download one.")

    def start_download(self):
        model = self.dl_model_var.get().strip()
        if not model:
            return
            
        self.dl_btn.config(state="disabled")
        self.dl_progress["value"] = 0
        self.dl_status_var.set("Connecting...")
        threading.Thread(target=download_model_thread, args=(model,), daemon=True).start()

    def send_message(self):
        prompt = self.input_text.get("1.0", "end").strip()
        model = self.active_model_var.get()
        
        if not prompt: return
        if not model:
            messagebox.showwarning("No Model", "Please select or download a model first.")
            return

        self._append_chat("You", prompt)
        self._append_chat(model, "", newline=False) # Prepare for streaming response
        
        self.input_text.delete("1.0", "end")
        self.send_btn.config(state="disabled")
        self.status_var.set("Generating response...")
        
        threading.Thread(target=chat_thread, args=(model, prompt), daemon=True).start()

    def on_enter_key(self, event):
        self.send_message()
        return "break"  # Prevents tkinter from adding a newline

    def on_shift_enter(self, event):
        self.input_text.insert(tk.INSERT, "\n")
        return "break"

    def _append_chat(self, who, text, newline=True):
        self.chat_box.configure(state="normal")
        if who:
            self.chat_box.insert("end", f"\n\n{who}: ")
        self.chat_box.insert("end", text)
        if newline:
            self.chat_box.insert("end", "")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _append_chat_stream(self, text):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    # --- Queue Processor ---
    def poll_queue(self):
        while True:
            try:
                msg = _gui_q.get_nowait()
                m_type = msg.get("type")

                if m_type == "status":
                    self.status_var.set(msg["msg"])
                
                elif m_type == "refresh_models":
                    self.refresh_installed_models()
                
                # --- Download Events ---
                elif m_type == "download_status":
                    self.dl_status_var.set(msg["status"])
                elif m_type == "download_progress":
                    pct = msg["percent"]
                    self.dl_progress["value"] = pct
                    self.dl_status_var.set(f"{msg['status']} ({pct}%)")
                elif m_type == "download_done":
                    self.dl_progress["value"] = 100
                    self.dl_status_var.set("✅ Download complete!")
                    self.dl_btn.config(state="normal")
                    self.status_var.set(f"✅ Downloaded {msg['model']}")
                    self.refresh_installed_models()
                elif m_type == "download_error":
                    self.dl_status_var.set("❌ Download failed.")
                    messagebox.showerror("Download Error", msg["error"])
                    self.dl_btn.config(state="normal")

                # --- Chat Events ---
                elif m_type == "chat_stream":
                    self._append_chat_stream(msg["content"])
                elif m_type == "chat_done":
                    self.send_btn.config(state="normal")
                    self.status_var.set("✅ Ready")
                elif m_type == "chat_error":
                    self._append_chat_stream(f"\n[Error: {msg['error']}]")
                    self.send_btn.config(state="normal")
                    self.status_var.set("❌ Chat error")

            except queue.Empty:
                break
                
        self.after(50, self.poll_queue)

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
