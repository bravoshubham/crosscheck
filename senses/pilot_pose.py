"""Pilot posture from SAM 3D Body keypoints (opt-in pilot watch camera).

Two measures, both relative to the pilot's own body, so they do not depend on where the camera
is: neck flexion (head drooping forward onto the chest) and how high the wrists are above the
hips (hands fallen off the controls into the lap). Thresholds were chosen on one generated image
set and tested on three held-out sets (eval/pilot_watch/).
A posture flag is never an alert by itself: Crosscheck first asks the pilot, and alerts only if
nobody answers.
"""
import numpy as np

# MHR70 keypoint indices (sam_3d_body/metadata/mhr70.py)
NOSE, L_EAR, R_EAR, L_SH, R_SH, L_HIP, R_HIP, R_WRIST, L_WRIST, NECK = 0, 3, 4, 5, 6, 9, 10, 41, 62, 69
# Each threshold is the midpoint between set a's normal and slumped values (the first set a images,
# now eval/pilot_watch/images_before_remake/a; the remade images were not used to tune them). A first, eyeballed
# WRISTS_MIN of 0.40 flagged 3 of 6 normal held-out images (sets b-d); after that change sets
# b-d are no longer a clean test, so sets e-f were made afterwards to test these values.
NECK_FLEX_MAX = 40.0   # degrees; set a: upright 19.6, reading a checklist 29.2, slumped forward 52.1
WRISTS_MIN = 0.31      # wrist height above the hips in torso lengths; set a: awake 0.40-0.44, hands in lap 0.20-0.23


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def posture(keypoints_3d) -> dict:
    k = np.asarray(keypoints_3d, dtype=float)
    ears, shoulders, hips = (k[L_EAR] + k[R_EAR]) / 2, (k[L_SH] + k[R_SH]) / 2, (k[L_HIP] + k[R_HIP]) / 2
    torso = shoulders - hips
    neck_flex = _angle(ears - k[NECK], torso)
    up = torso / np.linalg.norm(torso)
    wrists = ((k[L_WRIST] + k[R_WRIST]) / 2 - hips) @ up / np.linalg.norm(torso)
    reasons = []
    if neck_flex > NECK_FLEX_MAX:
        reasons.append("head dropped")
    if wrists < WRISTS_MIN:
        reasons.append("hands off the controls")
    return {"neck_flex": round(neck_flex, 1), "wrists": round(float(wrists), 2), "slumped": bool(reasons), "reasons": reasons}
