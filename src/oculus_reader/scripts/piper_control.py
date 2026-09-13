#!/usr/bin/env python3
"""Backward-compatible import for the old Piper controller name.

New code should import :class:`robot_control.RobotController` directly.  The
alias is kept so existing launch files and downstream scripts continue to
work while allowing ``~robot_backend`` to select RM75, SDK, simulator or
legacy joint-state control.
"""

from robot_control import PIPER, RobotController

__all__ = ["PIPER", "RobotController"]
