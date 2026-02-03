# egester/note_writer_api.py
import os
from fastapi import FastAPI, HTTPException
from .markdown_file_writer import MarkdownFileWriter
from utils.git_handler import GitHandler

DATA_DIR = "/app/data"
MARKDOWN_DIR = os.path.join(DATA_DIR, "markdown")
state_dir = "/app/state"
os.makedirs(state_dir, exist_ok=True)

app = FastAPI()

markdown_file_writer = MarkdownFileWriter()
git_handler = GitHandler(
        repo_dir=MARKDOWN_DIR,
        state_file=os.path.join(state_dir, ".git_processing_state.json") # Add state file although it is not needed
    )

@app.post("/write_note")
async def write_note(file_name: str, content: str, append: bool = False):
    """API endpoint to write or append a markdown note and commit to Git."""

    relative_file_path = os.path.join("ai_notes", file_name)

    file_path = os.path.join(MARKDOWN_DIR, relative_file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


    markdown_file_writer.save_file(
        content=content,
        file_path=file_path,
        append=append
    )

    if append is False:
        commit_message = f"AI created {relative_file_path} (markdown)"
    else:
        commit_message = f"AI updated {relative_file_path} (markdown)"


    if git_handler.verify_git_installed() and git_handler.is_git_repo():
        try:
            git_handler.git_add_and_commit(file_path=file_path, commit_message=commit_message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Git operation failed: {str(e)}")