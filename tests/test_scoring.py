import glob

from PIL import Image

from eval.scoring import _bright


def test_non_cockpit_frame_is_caught(tmp_path):
    # the failure seen live: the Claude window in front of X-Plane, captured as a white page
    page = tmp_path / "page.jpg"
    Image.new("RGB", (1940, 970), (250, 250, 250)).save(page)
    assert _bright(page)
    cockpit = sorted(glob.glob("replay/altitude_bust/frames/*.jpg"))[10]
    assert not _bright(cockpit)
