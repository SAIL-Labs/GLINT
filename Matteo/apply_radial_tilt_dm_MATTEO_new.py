"""
apply_radial_tilt_dm.py

Standalone SCExAO/GLINT helper script.

Purpose
-------
Apply a radial parking tip/tilt pattern to the Boston HEX111 DM segments 0..36.

This follows the same bench style as the working piston-scan scripts:
    sys.path.append('/home/scexao/glint/control-code/')
    import shmDMcontrol
    dm = shmDMcontrol.DM()
    dm.set_segment(segment, piston, tip, tilt)

Segment-number convention
-------------------------
The HEX111 37 segments are assumed to be displayed as a 3-ring hexagon with
segment numbers increasing top-to-bottom inside each column, and columns ordered
right-to-left, as used in the previous GLINT scripts/displays.

Tip/tilt calibration used
-------------------------
You provided two empirical directions:
    (tip, tilt) = (4, 0)   sends the beam along display direction (0, -1)
    (tip, tilt) = (-3, 5)  sends the beam 30 deg clockwise from negative x

The script builds a linear map from DM tip/tilt commands to beam direction,
then inverts that map to find the tip/tilt command that sends each segment
radially outward from the DM centre.

If the observed pattern is mirrored, flip CLOCKWISE_FROM_MINUS_X_SIGN.
If the beams go inward instead of outward, set RADIAL_DIRECTION = 'inward'.
"""

import sys
sys.path.append('/home/scexao/glint/control-code/')

import time
import numpy as np

import shmDMcontrol


# =============================================================================
# USER PARAMETERS
# =============================================================================

# Overall radial command amplitude in the same command units as your usual
# tip/tilt values. This is the approximate norm of the generated (tip, tilt)
# vector, not necessarily the exact beam-angle amplitude.
RADIAL_TILT_AMPLITUDE = 4.0

# Choose 'outward' to send each segment away from the centre, or 'inward' to
# send it toward the centre.
RADIAL_DIRECTION = 'outward'   # 'outward' or 'inward'

# Piston value applied to every segment while setting the radial tilt.
PISTON_VALUE = 0

# Segments that should stay injected into GLINT. These are kept flat in
# tip/tilt while all other segments receive the radial parking pattern.
ZERO_TIPTILT_SEGMENTS = [11, 20, 31]

# Centre segment behaviour. Usually keep the centre flat because its radial
# direction is undefined.
CENTRE_SEGMENT_TIPTILT = (-3, 5.0)

# Empirical calibration points, from your bench observation.
CMD_1 = (4.0, 0.0)
BEAM_DIRECTION_1_XY = (0.0, -1.0)   # beam goes down in the display plane

CMD_2 = (-3.0, 5.0)

# For a standard Oxy display with +y upward, "30 deg clockwise from -x" is 210 deg.
# If the resulting radial pattern is mirrored on the camera/display, change this
# from +1.0 to -1.0.
CLOCKWISE_FROM_MINUS_X_SIGN = +1.0
ANGLE_FROM_MINUS_X_DEG = 30.0

# Timing between segment commands.
DM_SETTLE_SECONDS = 0.001

# Print the generated table before applying it.
PRINT_TABLE = True

# Safety switch. Leave True to actually command the DM.
# Set False to only print the generated commands.
APPLY_TO_DM = True


# =============================================================================
# Geometry and calibration helpers
# =============================================================================

def hex37_segment_coordinates():
    """
    Return display-plane coordinates for segments 0..36.

    Geometry: 3-ring hexagonal lattice, sorted column-wise:
        right-to-left columns, and top-to-bottom within each column.

    The absolute scale is irrelevant for radial directions; only relative
    segment positions matter.
    """
    coords = []

    # Axial coordinates q, r for a radius-3 hexagon.
    # Convert to display-like 2D coordinates with pointy-top hex packing.
    radius = 3
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= radius:
                x = np.sqrt(3.0) * (q + 0.5 * r)
                y = 1.5 * r
                coords.append((x, y))

    # Previous display convention: numbers go top-to-bottom and right-to-left.
    # So sort by x descending, then y descending.
    coords_sorted = sorted(coords, key=lambda p: (-p[0], -p[1]))

    return {seg: np.array(coords_sorted[seg], dtype=float) for seg in range(37)}


def unit_vector_from_angle_deg(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)


def build_tiptilt_to_beam_matrix():
    """
    Build linear map M such that:
        beam_xy = M @ [tip, tilt]
    using the two empirical calibration vectors.
    """
    cmd1 = np.array(CMD_1, dtype=float)
    cmd2 = np.array(CMD_2, dtype=float)

    beam1 = np.array(BEAM_DIRECTION_1_XY, dtype=float)
    beam1 = beam1 / np.linalg.norm(beam1)

    # Negative x is 180 deg. Clockwise from negative x is +30 deg if using
    # the usual mathematical convention with +y upward and clockwise represented
    # here by the chosen empirical sign. Flip sign if mirrored.
    angle2 = 180.0 + CLOCKWISE_FROM_MINUS_X_SIGN * ANGLE_FROM_MINUS_X_DEG
    beam2 = unit_vector_from_angle_deg(angle2)

    command_matrix = np.column_stack([cmd1, cmd2])
    beam_matrix = np.column_stack([beam1, beam2])

    if abs(np.linalg.det(command_matrix)) < 1e-12:
        raise ValueError("Tip/tilt calibration commands are linearly dependent.")

    return beam_matrix @ np.linalg.inv(command_matrix)


def radial_tiptilt_commands():
    """
    Generate {segment: (tip, tilt)} for a radial beam pattern.
    """
    coords = hex37_segment_coordinates()
    centre = np.mean(np.array(list(coords.values())), axis=0)

    tiptilt_to_beam = build_tiptilt_to_beam_matrix()
    beam_to_tiptilt = np.linalg.inv(tiptilt_to_beam)

    sign = +1.0 if RADIAL_DIRECTION.lower() == 'outward' else -1.0
    if RADIAL_DIRECTION.lower() not in ['outward', 'inward']:
        raise ValueError("RADIAL_DIRECTION must be 'outward' or 'inward'.")

    commands = {}
    for seg in range(37):
        radial_xy = coords[seg] - centre
        norm = np.linalg.norm(radial_xy)

        if norm < 1e-12:
            commands[seg] = tuple(float(v) for v in CENTRE_SEGMENT_TIPTILT)
            continue

        desired_beam_xy = sign * radial_xy / norm
        raw_tiptilt = beam_to_tiptilt @ desired_beam_xy

        raw_norm = np.linalg.norm(raw_tiptilt)
        if raw_norm < 1e-12:
            tiptilt = np.array([0.0, 0.0])
        else:
            tiptilt = RADIAL_TILT_AMPLITUDE * raw_tiptilt / raw_norm

        commands[seg] = (float(tiptilt[0]), float(tiptilt[1]))

    # Force injected segments to have zero tip/tilt. This overrides the
    # radial pattern above, but keeps the same piston value when applied.
    for seg in ZERO_TIPTILT_SEGMENTS:
        if seg not in commands:
            raise ValueError(f"Invalid segment in ZERO_TIPTILT_SEGMENTS: {seg}")
        commands[seg] = (0.0, 0.0)

    return commands, coords


def print_command_table(commands, coords):
    print("\nRadial DM tip/tilt commands")
    print(f"RADIAL_DIRECTION = {RADIAL_DIRECTION}")
    print(f"RADIAL_TILT_AMPLITUDE = {RADIAL_TILT_AMPLITUDE}")
    print(f"CLOCKWISE_FROM_MINUS_X_SIGN = {CLOCKWISE_FROM_MINUS_X_SIGN}")
    print(f"ZERO_TIPTILT_SEGMENTS = {ZERO_TIPTILT_SEGMENTS}")
    print("\nSEG   X_DISPLAY   Y_DISPLAY        TIP        TILT")
    print("---------------------------------------------------")
    for seg in range(37):
        x, y = coords[seg]
        tip, tilt = commands[seg]
        print(f"{seg:3d} {x:11.5f} {y:11.5f} {tip:10.5f} {tilt:10.5f}")


def apply_to_dm(commands):
    dm = shmDMcontrol.DM()

    for seg in range(37):
        tip, tilt = commands[seg]
        dm.set_segment(int(seg), float(PISTON_VALUE), float(tip), float(tilt))
        time.sleep(DM_SETTLE_SECONDS)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    commands, coords = radial_tiptilt_commands()

    if PRINT_TABLE:
        print_command_table(commands, coords)

    if APPLY_TO_DM:
        print("\nApplying radial tip/tilt pattern to DM...")
        apply_to_dm(commands)
        print("Done.")
    else:
        print("\nAPPLY_TO_DM is False: commands were printed but not sent to the DM.")
