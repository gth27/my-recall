# My Recall

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red?style=flat)
![Postgres](https://img.shields.io/badge/SQL-PostgreSQL-336791?style=flat&logo=postgresql)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

**A privacy-first, open-source alternative to Windows Recall, built specifically for Linux power users.**

My Recall records your digital activity, indexes the text and visual concepts using local AI, and allows you to search your history using natural language. It is designed to run locally on your hardware, ensuring your data never leaves your machine.

---

### Note
> **This project was "vibe coded."**
> I built this tool primarily for my own personal use on Arch Linux (Hyprland). It solves *my* problems and fits *my* workflow. While I've engineered it to be robust (using Docker/Microservices), I am sharing it as-is. It might require some tweaking to fit your specific setup. 
>
> If you find it useful, great! If you want to improve it, pull requests are welcome.

---

## Key Features

* **Hybrid Search Engine:** Switch between modes to optimize your search:
    * **Exact Text:** Uses SQL to find specific code snippets, error logs, or messages.
    * **Visual AI:** Uses Vector Search (CLIP) to find concepts like *"video about cats"* or *"red website"* even if the text isn't present.
* **Linux Optimized:** Uses `grim` for lightweight, low-latency screen capture.
* **Privacy Focused:**
    * **Incognito Blocker:** Automatically pauses recording when "Private" or "Incognito" windows are active.
    * **Local Processing:** Uses PaddleOCR and CLIP running entirely on your CPU.
* **Microservice Architecture:** Built on a robust stack of Docker containers (Backend, Qdrant, Postgres, Frontend).

---

## Architecture

My Recall uses a scalable microservices architecture:

1.  **Watcher (Host Machine):** A lightweight Python script that monitors window states and captures screenshots. It handles the "Incognito" checks and writes to a shared volume.
2.  **Backend (Docker):** A worker service that processes incoming images using **PaddleOCR** (for text) and **CLIP** (for visual embeddings).
3.  **Databases:**
    * **Qdrant:** Stores vector embeddings for semantic search.
    * **PostgreSQL:** Stores metadata, timestamps, and raw text for exact keyword search.
4.  **Frontend:** A Streamlit dashboard for browsing, searching, and managing your data.

---

## Quick Start

### Prerequisites
* **Linux** (Arch/Hyprland recommended, adaptable to others)
* **Docker** & **Docker Compose**
* **Python 3.10+**
* `grim` (screenshot utility)

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/linux-recall.git](https://github.com/YOUR_USERNAME/linux-recall.git)
    cd linux-recall
    ```

2.  **Set up the Watcher (Host)**
    The watcher runs on your host machine to access the display server directly.
    ```bash
    cd watcher
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
    ```

3.  **Launch the System**
    We provide a single script to start the Docker engine and the local watcher in the background.
    ```bash
    ./start.sh
    ```

4.  **Access the Dashboard**
    Open your browser to: **`http://localhost:8501`**

---

## Usage

### **Search Modes**
Use the sidebar to toggle between search strategies:
* **Hybrid (Recommended):** Combines both text and visual results.
* **Text Only:** Faster, lightweight mode that only matches exact words (0% AI usage).
* **Visual Only:** Finds images based on "vibe" or visual description.

### **Controls**
* **Pause/Resume:** Click the button in the sidebar (or run the script) to temporarily stop recording.
* **Danger Zone:** Safely wipe all database entries and screenshots if you want a fresh start.

### **Stopping the System**
To shut down the watcher and all Docker containers:
```bash
./stop.sh

```

---

## Configuration

Edit `watcher/config.yaml` to tweak capture behavior:

```yaml
storage_path: "../data/inbox"
capture_interval: 2.0       # Seconds between screenshots
similarity_threshold: 3     # Higher = less strict deduplication
window_blacklist:           # Stop recording if window title contains:
  - "Incognito"
  - "Private Browsing"
  - "Bitwarden"
  - "1Password"

```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
