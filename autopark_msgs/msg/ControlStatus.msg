# Path tracking controller status (published at the control rate while a path is active).
std_msgs/Header header
uint32 plan_id
string state              # idle | align | track | stop | done | aborted
int32 segment             # current segment of constant direction (0-based)
int32 segments
int8 direction            # +1 forward, -1 reverse
float32 lateral_error     # m, reference point from its path (front axle forward, rear axle reverse)
float32 heading_error     # rad
float32 remaining         # m to the end of the current segment
float32 max_lateral_error # m, over the path so far
