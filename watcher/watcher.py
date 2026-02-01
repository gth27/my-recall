import time
import os
import subprocess
import logging
import yaml
import imagehash
from PIL import Image
from datetime import datetime

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- PATH ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Helper to load config relative to the script
def load_config():
    config_path = os.path.join(SCRIPT_DIR, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# 3. Resolve the storage path relative to the script
STORAGE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, CONFIG['storage_path']))
PAUSE_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/recall.pause"))

# Ensure storage exists
os.makedirs(STORAGE_PATH, exist_ok=True)

class RecallWatcher:
    def __init__(self):
        self.last_hash = None

    def is_paused(self):
        if os.path.exists(PAUSE_FILE):
            return True
        return False

    def get_active_window_title(self):
        try:
            output = subprocess.check_output(
                "hyprctl activewindow -j | grep '\"title\"'", 
                shell=True
            ).decode().strip()
            if ":" in output:
                return output.split(":", 1)[1].strip().strip('",')
            return ""
        except Exception:
            return "Unknown"

    def is_safe_to_capture(self):
        title = self.get_active_window_title()
        for blocked_word in CONFIG['window_blacklist']:
            if blocked_word.lower() in title.lower():
                logger.info(f"Privacy Block: Active window '{title}' matches blacklist.")
                return False
        return True

    def capture(self):
        if self.is_paused():
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        temp_filename = "temp_capture.png"
        temp_path = os.path.join(STORAGE_PATH, temp_filename)

        try:
            if not self.is_safe_to_capture():
                return

            subprocess.run(
                ["grim", "-t", "png", "-l", "0", temp_path], 
                check=True, 
                stderr=subprocess.DEVNULL
            )

            img = Image.open(temp_path)
            current_hash = imagehash.phash(img)

            if self.last_hash is not None:
                diff = current_hash - self.last_hash
                if diff < CONFIG['similarity_threshold']:
                    return

            final_filename = f"{timestamp}.png"
            final_path = os.path.join(STORAGE_PATH, final_filename)
            os.rename(temp_path, final_path)
            
            self.last_hash = current_hash
            logger.info(f"Saved: {final_filename}")

        except subprocess.CalledProcessError:
            logger.error("Capture failed. Is Hyprland running?")
        except Exception as e:
            logger.error(f"Error: {e}")

    def run(self):
        logger.info("Recall Watcher Started.")
        logger.info(f"Saving to: {STORAGE_PATH}")
        
        try:
            while True:
                self.capture()
                time.sleep(CONFIG['capture_interval'])
        except KeyboardInterrupt:
            logger.info("Watcher stopped by user.")

if __name__ == "__main__":
    watcher = RecallWatcher()
    watcher.run()
