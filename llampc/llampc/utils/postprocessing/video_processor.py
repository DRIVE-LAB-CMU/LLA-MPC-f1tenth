import os
import re
import subprocess
import uuid
import threading
import concurrent.futures
from collections import defaultdict

# --- CONFIGURATION ---
ROOT_DIRECTORY = r"C:\Users\henry\Videos\ws\LLA-MPC JUST VIDEOS"
DELETE_ORIGINALS_AFTER_STITCH = False 
MAX_WORKERS = 3
# ---------------------

# A lock to ensure console print statements don't overlap from different threads
print_lock = threading.Lock()

def safe_print(message):
    with print_lock:
        print(message)

def get_base_and_index(filename):
    """Parses filenames to separate the base name and the split index."""
    match = re.match(r'^(.*?)(?: \((\d+)\))?\.avi$', filename, re.IGNORECASE)
    if match:
        base_name = match.group(1)
        index = int(match.group(2)) if match.group(2) else 0
        return base_name, index
    return None, None

def process_video_group(dirpath, base, files):
    """Processes a single group of videos (either renaming or stitching)."""
    # Sort by index: base.avi (0), then (1), then (2)...
    files.sort(key=lambda x: x[0])
    
    complete_filename = f"{base} (complete).avi"
    complete_filepath = os.path.join(dirpath, complete_filename)
    
    if len(files) == 1:
        # 1. ONLY ONE VIDEO: Just rename it
        old_filepath = os.path.join(dirpath, files[0][1])
        safe_print(f"[RENAME] {files[0][1]}\n      -> {complete_filename}")
        os.rename(old_filepath, complete_filepath)
        
    else:
        # 2. MULTIPLE VIDEOS: Stitch them together
        safe_print(f"[STITCH] {len(files)} parts into:\n      -> {complete_filename}")
        
        unique_id = uuid.uuid4().hex[:8]
        list_file_name = f"ffmpeg_concat_{unique_id}.txt"
        list_file_path = os.path.join(dirpath, list_file_name)
        
        with open(list_file_path, "w", encoding="utf-8") as lf:
            for _, f in files:
                lf.write(f"file '{f}'\n")
        
        cmd = [
            "ffmpeg", 
            "-f", "concat", 
            "-safe", "0", 
            "-i", list_file_name, 
            "-c", "copy", 
            complete_filename
        ]
        
        try:
            subprocess.run(
                cmd, 
                cwd=dirpath, 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            safe_print(f"[DONE]   Successfully created {complete_filename}")
            
            if os.path.exists(list_file_path):
                os.remove(list_file_path)
            
            if DELETE_ORIGINALS_AFTER_STITCH:
                for _, f in files:
                    os.remove(os.path.join(dirpath, f))
                    
        except subprocess.CalledProcessError:
            safe_print(f"[ERROR]  Failed to stitch {base}. Is FFmpeg installed?")
        except FileNotFoundError:
            safe_print(f"[ERROR]  FFmpeg not found on system PATH.")
        finally:
            if os.path.exists(list_file_path):
                os.remove(list_file_path)

def process_videos_concurrently(root_dir):
    tasks = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        video_groups = defaultdict(list)
        completed_bases = set()
        
        # --- PASS 1: Identify what is already complete ---
        for f in filenames:
            if f.lower().endswith('(complete).avi'):
                # Extract the base name from the complete file
                match = re.match(r'^(.*?) \(complete\)\.avi$', f, re.IGNORECASE)
                if match:
                    completed_bases.add(match.group(1).lower())
        
        # --- PASS 2: Group files, skipping the completed bases ---
        for f in filenames:
            if f.lower().endswith('.avi'):
                if f.lower().endswith('(complete).avi'):
                    continue
                    
                base, index = get_base_and_index(f)
                
                # Only add to our processing queue if the base name is not in our completed list
                if base and base.lower() not in completed_bases:
                    video_groups[base].append((index, f))
        
        # Append each pending group as a separate task
        for base, files in video_groups.items():
            tasks.append((dirpath, base, files))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No pending videos found. Everything is already complete!")
        return

    print(f"Found {total_tasks} pending video groups to process.")
    print(f"Starting ThreadPoolExecutor with {MAX_WORKERS} workers...\n" + "-"*50)

    completed_tasks = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_video_group, dirpath, base, files) 
                   for dirpath, base, files in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                safe_print(f"[CRITICAL] A worker thread generated an exception: {exc}")
            
            completed_tasks += 1
            progress_percent = (completed_tasks / total_tasks) * 100
            safe_print(f"\n---> [PROGRESS] {completed_tasks}/{total_tasks} ({progress_percent:.1f}%) video groups processed <---")

if __name__ == "__main__":
    process_videos_concurrently(ROOT_DIRECTORY)
    print("\n" + "-"*50 + "\nAll processing complete!")