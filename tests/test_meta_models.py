from brain.explain import context
from brain.ledger import Fact, Ledger
from brain.rules import Flag
from senses.sam31 import iou, parse_boxes


def test_sam31_answer_is_parsed_into_boxes():
    # the shape of a real answer: instance marker, box, mask (truncated)
    text = "<0f>0<|box;x1=1037;y1=117;x2=1828;y2=881;w=1940;h=970|><|mask;x=0;y=0;data=765,792,~!!|><1f>0<|box;x1=110;y1=117;x2=890;y2=877;w=1940;h=970|>"
    assert parse_boxes(text) == [(1037, 117, 1828, 881), (110, 117, 890, 877)]


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_explanation_gets_the_recent_facts_and_only_moving_panel_values():
    led = Ledger()
    led.add(Fact("said", "clearance.altitude", 11000, 10.0, "atc"))
    for t, alt in ((11, 9000), (12, 9010), (13, 9200)):  # 10 ft is not a move; 200 ft is
        led.add(Fact("doing", "altitude", alt, float(t)))
    led.add(Fact("set", "mcp.altitude", 12000, 12.0))
    led.add(Fact("said", "clearance.altitude", 8000, -100.0, "atc"))  # outside the window
    flag = Flag("clearance_vs_set", "altitude", 11000, 12000, 12.0, "Cleared one-one thousand, set one-two thousand.")
    p = context(led, flag, 14.0)
    assert "clearance.altitude (atc) = 11000" in p and "mcp.altitude = 12000" in p
    assert "= 9010" not in p and "= 9200" in p and "= 8000" not in p
