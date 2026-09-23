"""
Part 5 — Swing Phase Detector

A plain state machine (no ML) that watches hand-height and shoulder-rotation
velocity to decide which swing phase the golfer is currently in:
READY -> ADDRESS -> BACKSWING -> TOP -> DOWNSWING -> IMPACT -> FOLLOW_THROUGH -> READY
"""

from __future__ import annotations

from . import config
from .skeleton import midpoint


def get_hand_y(image_landmarks) -> float:
    return (image_landmarks[15]["y"] + image_landmarks[16]["y"]) / 2


def get_body_unit(image_landmarks) -> float:
    mid_shoulder = midpoint(image_landmarks[11], image_landmarks[12])
    mid_hip = midpoint(image_landmarks[23], image_landmarks[24])
    return abs(mid_hip["y"] - mid_shoulder["y"]) or 1.0


def is_hands_low(image_landmarks) -> bool:
    mid_hip = midpoint(image_landmarks[23], image_landmarks[24])
    return get_hand_y(image_landmarks) >= mid_hip["y"] - config.HANDS_LOW_RATIO * get_body_unit(image_landmarks)


def _angle_delta(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


class AngleBasedSwingDetector:
    """Frame-by-frame swing phase state machine.

    Call :meth:`update` once per frame with the current shoulder/hip
    rotation angles and hand height; it returns ``(phase, events)`` where
    ``events`` is a list of ``(event_kind, frame_idx, swing_number)`` tuples
    describing anything notable that happened on this frame (a swing
    starting, reaching the top, impacting, ending, being cancelled, etc.).
    """

    def __init__(self, body_unit: float, fps_scale: float = 1.0):
        self.body_unit = body_unit
        s = max(0.34, fps_scale)
        self.confirm = max(1, round(config.CONFIRM_FRAMES * s))
        self.reset_hold = max(1, round(config.RESET_HOLD_FRAMES * s))
        self.calib_still = max(1, round(config.CALIB_STILL_FRAMES * s))
        self.calib_max = max(self.calib_still + 1, round(config.CALIB_MAX_FRAMES * s))
        self.impact_hold = max(2, round(2 * s))
        self.timeouts = {k: round(v * s) for k, v in config.STATE_TIMEOUT_FRAMES.items()}

        self.phase = "READY"
        self._counters = {}
        self._state_frames = 0
        self._calib_frames = 0

        self.ref_shoulder_raw = None
        self.ref_hip_raw = None
        self.address_hand_y = None

        self._shoulder_ema = None
        self._hip_ema = None
        self._prev_ema = None
        self._prev_shoulder_raw = None
        self._prev_hand_y = None
        self.shoulder_vel = 0.0
        self.rel_signed = 0.0
        self._backswing_sign = 1.0
        self._peak_shoulder = 0.0
        self._impact_frame = None
        self.swing_count = 0
        self.completed_swings = 0

        self.timeline = []
        self._events = []

    # ---------- helpers ----------
    def _note(self, frame, msg):
        self.timeline.append((frame, msg))

    def _advance(self, key, condition, needed):
        self._counters[key] = self._counters.get(key, 0) + 1 if condition else 0
        return self._counters[key] >= needed

    def _to(self, frame, phase, reason=""):
        prev = self.phase
        self.phase = phase
        self._state_frames = 0
        self._counters.clear()
        if phase != prev:
            self._note(frame, f"{prev} -> {phase}" + (f" ({reason})" if reason else ""))
            if phase == "TOP":
                self._events.append(("top", frame, self.swing_count))
            elif phase == "DOWNSWING":
                self._events.append(("downswing", frame, self.swing_count))

    def _smooth(self, ema, new):
        return new if ema is None else ema + _angle_delta(new, ema) * config.ANGLE_EMA

    def hard_reset(self, frame_idx, reason=""):
        """Forces the machine back to READY — used on a body-tracking loss or a scene cut."""
        events = []
        if self.phase in ("BACKSWING", "TOP", "DOWNSWING"):
            events.append(("swing_abort", frame_idx, self.swing_count))
            self._note(frame_idx, f"swing #{self.swing_count} cut short ({reason or self.phase})")
        elif self.phase in ("IMPACT", "FOLLOW_THROUGH"):
            self.completed_swings += 1
            events.append(("swing_end", frame_idx, self.swing_count))
            self._note(frame_idx, f"swing #{self.swing_count} closed ({reason or self.phase})")
        self.phase = "READY"
        self._counters.clear()
        self._state_frames = 0
        self._calib_frames = 0
        self.ref_shoulder_raw = self.ref_hip_raw = None
        self.address_hand_y = None
        self._shoulder_ema = self._hip_ema = self._prev_ema = None
        self.shoulder_vel = 0.0
        self.rel_signed = 0.0
        self._peak_shoulder = 0.0
        return events

    def _enter_address(self, shoulder_raw, hip_raw, hand_y, frame_idx, events):
        self.ref_shoulder_raw = shoulder_raw
        self.ref_hip_raw = hip_raw
        self.address_hand_y = hand_y
        self._shoulder_ema = shoulder_raw
        self._hip_ema = hip_raw
        self._prev_ema = shoulder_raw
        self.shoulder_vel = 0.0
        self.rel_signed = 0.0
        self._to(frame_idx, "ADDRESS", "calibrated")
        events.append(("address_set", frame_idx, None))

    # ---------- main loop entry point ----------
    def update(self, shoulder_raw, hip_raw, hand_y, hands_low, frame_idx):
        events = []
        self._events = events

        # 0) Scene-cut detection = both a shoulder jump AND a hand jump (AND, not OR)
        if self._prev_shoulder_raw is not None:
            jump_deg = abs(_angle_delta(shoulder_raw, self._prev_shoulder_raw))
            jump_hand = (
                self._prev_hand_y is not None
                and abs(hand_y - self._prev_hand_y) >= config.CUT_HAND_JUMP_RATIO * self.body_unit
            )
            if jump_deg >= config.CUT_SHOULDER_JUMP_DEG and jump_hand:
                self._note(frame_idx, f"scene cut (jump {jump_deg:.0f} deg + hand jump)")
                events.extend(self.hard_reset(frame_idx, "scene cut"))
                events.append(("scene_reset", frame_idx, None))
                self._prev_shoulder_raw = shoulder_raw
                self._prev_hand_y = hand_y
                return self.phase, events
        self._prev_shoulder_raw = shoulder_raw
        self._prev_hand_y = hand_y

        self._shoulder_ema = self._smooth(self._shoulder_ema, shoulder_raw)
        self._hip_ema = self._smooth(self._hip_ema, hip_raw)
        if self._prev_ema is not None:
            self.shoulder_vel = _angle_delta(self._shoulder_ema, self._prev_ema)
        self._prev_ema = self._shoulder_ema
        if self.ref_shoulder_raw is not None:
            self.rel_signed = _angle_delta(self._shoulder_ema, self.ref_shoulder_raw)
        rel = abs(self.rel_signed)
        dir_vel = self.shoulder_vel * self._backswing_sign

        self._state_frames += 1

        # 1) READY: waiting for a still, hands-low pose to calibrate against
        if self.phase == "READY":
            self._calib_frames += 1
            still = abs(self.shoulder_vel) <= config.ANGULAR_STILL_DEG
            if self._advance("calib", still and hands_low, self.calib_still) or (
                self._calib_frames >= self.calib_max and hands_low
            ):
                self._enter_address(shoulder_raw, hip_raw, hand_y, frame_idx, events)

        # 2) ADDRESS -> BACKSWING
        elif self.phase == "ADDRESS":
            if self._advance("backswing", rel > config.BACKSWING_START_DEG, self.confirm):
                self._backswing_sign = 1.0 if self.rel_signed >= 0 else -1.0
                self._peak_shoulder = rel
                self.swing_count += 1
                self._to(frame_idx, "BACKSWING", f"swing #{self.swing_count} started")
                events.append(("swing_start", frame_idx, self.swing_count))

        # 3) BACKSWING -> TOP (or cancel, if it was just a waggle)
        elif self.phase == "BACKSWING":
            self._peak_shoulder = max(self._peak_shoulder, rel)
            top_now = config.TOP_MIN_DEG <= rel <= config.TOP_MAX_DEG and dir_vel <= config.TOP_PEAK_DROP_DEG
            if self._advance("top", top_now, self.confirm):
                self._to(frame_idx, "TOP", f"rotation {rel:.0f} deg")
            elif self._advance("cancel", rel < config.WAGGLE_CANCEL_DEG, self.confirm * 3):
                cancelled = self.swing_count
                self.swing_count -= 1
                self._to(frame_idx, "ADDRESS", "cancelled waggle")
                events.append(("swing_cancelled", frame_idx, cancelled))

        # 4) TOP -> DOWNSWING
        elif self.phase == "TOP":
            if self._advance("down", dir_vel <= -config.DOWNSWING_UNWIND_DEG, self.confirm):
                self._to(frame_idx, "DOWNSWING")

        # 5) DOWNSWING -> IMPACT
        elif self.phase == "DOWNSWING":
            by_angle = self._advance("impact_ang", rel <= config.IMPACT_ROT_MAX_DEG, self.confirm)
            by_hand = self.address_hand_y is not None and hand_y >= self.address_hand_y
            if by_angle or by_hand:
                self._impact_frame = frame_idx
                self._to(frame_idx, "IMPACT", "angle" if by_angle else "hand")
                events.append(("impact", frame_idx, self.swing_count))

        # 6) IMPACT: hold briefly
        elif self.phase == "IMPACT":
            if frame_idx - self._impact_frame >= self.impact_hold:
                self._to(frame_idx, "FOLLOW_THROUGH")

        # 7) FOLLOW_THROUGH: three ways out (settle, rebound into next swing, or re-address)
        elif self.phase == "FOLLOW_THROUGH":
            backswing_side = self.rel_signed * self._backswing_sign
            settled = rel <= config.RESET_RETURN_DEG and abs(self.shoulder_vel) <= config.ANGULAR_STILL_DEG
            if self._advance("reset", settled, self.reset_hold):
                self.completed_swings += 1
                self._to(frame_idx, "READY", f"swing #{self.swing_count} ended")
                events.append(("swing_end", frame_idx, self.swing_count))
            elif self._advance("rebound", backswing_side > config.BACKSWING_START_DEG, self.confirm):
                self.completed_swings += 1
                events.append(("swing_end", frame_idx, self.swing_count))
                self._backswing_sign = 1.0 if self.rel_signed >= 0 else -1.0
                self._peak_shoulder = rel
                self.swing_count += 1
                self._to(frame_idx, "BACKSWING", f"consecutive swing #{self.swing_count}")
                events.append(("swing_start", frame_idx, self.swing_count))
            elif self._advance(
                "readdress",
                hands_low and abs(self.shoulder_vel) <= config.ANGULAR_STILL_DEG,
                self.reset_hold * 2,
            ):
                self.completed_swings += 1
                events.append(("swing_end", frame_idx, self.swing_count))
                self._to(frame_idx, "READY", "returned to address (stale reference)")

        # Timeout safety net
        limit = self.timeouts.get(self.phase)
        if limit is not None and self._state_frames >= limit:
            if self.phase in ("BACKSWING", "TOP", "DOWNSWING"):
                events.append(("swing_abort", frame_idx, self.swing_count))
                self._note(frame_idx, f"swing #{self.swing_count} cut short (timeout {self.phase})")
            elif self.phase == "FOLLOW_THROUGH":
                self.completed_swings += 1
                events.append(("swing_end", frame_idx, self.swing_count))
                self._note(frame_idx, f"swing #{self.swing_count} closed (timeout follow-through)")
            self._to(frame_idx, "READY", "timeout")
            self._calib_frames = 0

        return self.phase, events
