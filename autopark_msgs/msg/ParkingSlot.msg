# One perpendicular parking slot, expressed in the frame of the enclosing header.
#
# Geometry is measured between line centres. The entrance is the open end of the slot.
# heading points INTO the slot (from the entrance towards the back line).
#
#   corners[3] ------------- corners[2]      back line
#       |                        |
#       |      ^ heading         |
#       |      |                 |
#   corners[0]   entrance    corners[1]      open end (no line)
#
# corners[0] = entrance left, corners[1] = entrance right (seen from the aisle, looking into
# the slot), corners[2] = back right, corners[3] = back left.

int32 id                      # ground-truth slot id, or track id (/slots/tracked); -1 for raw detections
geometry_msgs/Point[4] corners
geometry_msgs/Pose2D entrance # centre of the entrance line + heading into the slot
float64 width                 # m, between side-line centres
float64 depth                 # m, entrance to back-line centre
bool occupied
float32 vacancy               # probability that the slot is free (0..1); -1 if unknown
float32 confidence            # 1.0 for ground truth
float64[9] covariance         # row-major 3x3 of (entrance.x, entrance.y, entrance.theta); zeros if unknown
