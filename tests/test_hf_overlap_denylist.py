from scripts.build_hf_overlap_denylist import records


def test_records_paginates_and_never_returns_plaintext():
    def fetch(path, params):
        if path == "splits":
            return {"splits": [{"config": "default", "split": "train"}]}
        if params["offset"] == 0:
            return {"num_rows_total": 2, "rows": [{"row": {"video_id": "a", "text": "කොළඹ"}}]}
        return {"num_rows_total": 2, "rows": [{"row": {"video_id": "b", "text": "රාජගිරිය"}}]}

    result = list(records("owner/dataset", fetch))
    assert [row["video_id"] for row in result] == ["a", "b"]
    assert all("text" not in row for row in result)
    assert all(len(row["text_sha256"]) == 64 for row in result)
