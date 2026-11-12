import pytest

from sensegraph.real_mapping import chartqa_pixel_annotations, vistext_local_nodes


def annotation(chart_type="h_bar"):
    return {"type": chart_type, "general_figure_info": {"figure_info": {"bbox": {"x": 10, "y": 10, "w": 70, "h": 70}}},
            "models": [{"name": "S", "x": ["A", "B"], "y": [2, 4],
                        "bboxes": [{"x": 10, "y": 10, "w": 20, "h": 5}, {"x": 10, "y": 20, "w": 40, "h": 5}]}]}


def test_chartqa_xywh_conversion_and_source_membership():
    result = chartqa_pixel_annotations(annotation(), (100, 100))
    mark = next(n for n in result["nodes"] if n["id"] == "mark:0:0")
    assert mark["bbox"] == [10, 10, 30, 15]
    assert mark["source_value"] == 2
    assert {"source": "mark:0:0", "target": "series:0", "relation": "belongs_to", "provenance": "source models grouping"} in result["edges"]


def test_line_segments_do_not_get_point_values_or_fake_padding():
    raw = annotation("line")
    raw["models"][0]["bboxes"] = [{"x": 10, "y": 20, "w": 40, "h": 0}]
    result = chartqa_pixel_annotations(raw, (100, 100))
    mark = next(n for n in result["nodes"] if n["id"] == "mark:0:0")
    assert mark["type"] == "line_segment" and mark["numeric_value"] is None
    assert mark["data_endpoints"] == {"x": ["A", "B"], "y": [2, 4]}
    assert mark["bbox"] is None
    assert "zero-area" in result["issues"][0]["reason"]


def test_chartqa_ambiguous_and_out_of_range_boxes():
    with pytest.raises(ValueError, match="alignment"):
        chartqa_pixel_annotations(annotation("line"), (100, 100))
    result = chartqa_pixel_annotations(annotation(), (20, 20))
    assert result["issues"]
    assert all(node["bbox"] is None for node in result["nodes"])


def test_vistext_nested_group_translation_without_raster_invention():
    graph = {"marktype": "group", "items": [{"x": 10, "y": -20, "items": [
        {"marktype": "group", "items": [{"x": 2, "y": 5, "items": [
            {"marktype": "rect", "role": "mark", "items": [{"x": 3, "y": 4, "width": 8, "height": 9}]}]}]}]}]}
    node = vistext_local_nodes(graph)[0]
    assert node["point_chart_xy"] == [15, -11]
    assert node["bbox_chart_xyxy"] == [15, -11, 23, -2]
    assert node["bbox"] is None


def test_text_anchors_do_not_invent_font_bounds():
    graph = {"marktype": "text", "items": [{"x": 10, "y": 20, "text": "date", "fontSize": 12}]}
    node = vistext_local_nodes(graph)[0]
    assert node["point_chart_xy"] == [10, 20]
    assert node["bbox_chart_xyxy"] is None and node["bbox"] is None


def test_unsupported_group_rotation_is_not_projected_as_translation():
    graph = {"marktype": "group", "items": [{"x": 10, "y": 20, "angle": 30, "items": [
        {"marktype": "rect", "items": [{"x": 1, "y": 2, "width": 3, "height": 4}]}]}]}
    node = vistext_local_nodes(graph)[0]
    assert node["type"] == "unsupported_group_transform"
    assert node["point_chart_xy"] is None and node["bbox"] is None
