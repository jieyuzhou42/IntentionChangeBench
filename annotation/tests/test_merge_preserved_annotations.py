from annotation.merge_preserved_annotations import preserve_and_merge


def trajectory(instance_id, marker):
    return {"instance_id": instance_id, "marker": marker, "turns": []}


def test_preserve_and_merge_replaces_survivors_and_omits_deleted_first_items():
    old_source = [trajectory(f"id_{index}", "old") for index in range(5)]
    old_annotation = [
        trajectory("id_0", "annotated"),
        trajectory("id_2", "annotated"),
        trajectory("id_3", "unreviewed old tail"),
        trajectory("id_4", "unreviewed old tail"),
    ]
    new_dataset = [trajectory(f"id_{index}", "new") for index in range(5)]

    preserved, merged, manifest = preserve_and_merge(
        old_source, old_annotation, new_dataset, first_count=3
    )

    assert [item["instance_id"] for item in preserved] == ["id_0", "id_2"]
    assert [item["instance_id"] for item in merged] == ["id_0", "id_2", "id_3", "id_4"]
    assert [item["marker"] for item in merged] == ["annotated", "annotated", "new", "new"]
    assert manifest["deleted_instance_ids"] == ["id_1"]
    assert manifest["preserved_count"] == 2
    assert manifest["merged_count"] == 4
