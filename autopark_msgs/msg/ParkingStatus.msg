# Parking manager state.
std_msgs/Header header
string state              # idle | search | stopping | planning | executing | done | failed
int32 slot_id             # target track id (-1 = none yet)
uint32 plan_id
int32 plans               # plans computed for this manoeuvre (1 = planned once)
int32 corrections         # path corrections applied on the final approach
string message
