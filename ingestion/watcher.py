# ingestion/watcher.py
import time
from watchdog.events import FileSystemEventHandler
from .ingestion_service import MarkdownIngestionService

class MarkdownWatchHandler(FileSystemEventHandler):
    """Handles filesystem events and delegates to ingestion service."""
    
    def __init__(self, ingestion_service: MarkdownIngestionService, debounce_seconds: int = 5):
        super().__init__()
        self.ingestion_service = ingestion_service
        self.debounce_seconds = debounce_seconds
        self.last_processed = 0
    
    def on_modified(self, event):
        """Handle file modification events with immediate processing"""
        if event.is_directory or not event.src_path.endswith('.md'):
            return
        
        current_time = time.time()
        if current_time - self.last_processed > self.debounce_seconds:
            print(f"Detected modification: {event.src_path}")
            self.ingestion_service.process_single_file(event.src_path)
            self.last_processed = current_time
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory or not event.src_path.endswith('.md'):
            return
        
        print(f"Detected new file: {event.src_path}")
        self.ingestion_service.process_single_file(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if event.is_directory or not event.src_path.endswith('.md'):
            return
        
        print(f"Detected deleted file: {event.src_path}")
        self.ingestion_service.delete_file(event.src_path)
