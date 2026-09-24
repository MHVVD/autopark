# A planned parking manoeuvre: rear-axle (base_link) poses sampled every ~0.1 m, in the frame
# of the header (odom), with the gear for each sample.
#
# direction[i] / curvature[i] describe the motion that reaches poses[i] (poses[0] copies the
# first motion). A change of direction between i and i+1 is a cusp: the car stops at poses[i]
# and changes gear.

std_msgs/Header header
int32 slot_id                    # track id of the target slot (/slots/tracked)
geometry_msgs/Pose2D goal        # final rear-axle pose: centred in the slot, facing out
geometry_msgs/Pose2D[] poses
int8[] direction                 # +1 forward, -1 reverse
float32[] curvature              # 1/m, positive = turning left
float32 length                   # m
int32 gear_changes
float32 planning_time            # s
