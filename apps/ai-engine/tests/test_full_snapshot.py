import io
import tarfile
from unittest.mock import patch

from services.github_fetcher import fetch_repo_files


def test_snapshot_reads_all_matching_files_and_keeps_source_filter():
    archive = io.BytesIO()
    paths = [f"src/C{i}.java" for i in range(65)]
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in [*paths, "web/View.TSX", "web/node_modules/lib.ts", "build/C.java", "README.md"]:
            entry = tarfile.TarInfo(f"repo-commit/{path}")
            entry.size = 1
            tar.addfile(entry, io.BytesIO(b"x"))
    archive.seek(0)
    with patch("services.github_fetcher.urllib.request.urlopen", return_value=archive):
        files = list(fetch_repo_files("owner", "repo", "a" * 40))
    assert [f["file_path"] for f in files] == [*paths, "web/View.TSX"]
    assert files[-1]["language"] == "tsx"
