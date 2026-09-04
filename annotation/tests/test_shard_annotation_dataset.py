import pytest

from annotation.shard_annotation_dataset import build_shards


def trajectory(instance_id, marker):
    return {"instance_id": instance_id, "marker": marker, "turns": []}


def test_build_shards_aligns_source_to_annotation_order_and_size():
    source = [trajectory(f"id_{index}", "source") for index in range(7)]
    annotation = [trajectory(f"id_{index}", "annotation") for index in (0, 2, 3, 5, 6)]

    shards = build_shards(source, annotation, shard_size=2)

    assert [shard["count"] for shard in shards] == [2, 2, 1]
    assert [item["instance_id"] for item in shards[0]["source"]] == ["id_0", "id_2"]
    assert [item["marker"] for item in shards[0]["annotation"]] == ["annotation", "annotation"]
    assert shards[2]["first_instance_id"] == "id_6"


def test_build_shards_rejects_annotation_id_missing_from_source():
    with pytest.raises(ValueError, match="missing from source"):
        build_shards([trajectory("id_0", "source")], [trajectory("id_1", "annotation")], 60)
