from repositories import SQLiteRepository


def test_repository_persists_task_and_result(tmp_path):
    repository = SQLiteRepository(tmp_path / "db.sqlite3")
    repository.create_task(task_id="task-1", video_file_id="file-1", llm_enabled=False)
    repository.save_result("task-1", {"segments": [], "video_duration": 1}, enhanced=False, result_source="rule")

    task = repository.get_task("task-1")
    result = repository.get_result("task-1")

    assert task is not None
    assert task.task_id == "task-1"
    assert result["video_duration"] == 1
