#Mash up Speed, Orientation and Success

#Hyper Crude Prototype:
# cos between gradient and heading * y * distance (or so) for right direction

#Crude Prototype:
# x for reaching goal, 
# cos between gradient and heading * y * distance (or so) for right direction
# flat penalty for crashing (less for good driving)
# (soft?) cap gradient, so it doesnt dominate.

#Proper Prototype:
#Big reward for goal, big punishment for fatal end
#Limited reward for gradient chasing.
# Ensure Reward Hacking is not possible.

#Actual Idea:
#Alleviate Speed Penalty for Alignment, if success
#Alleviate Crash Penalty for Alignment, if crash.
#Probably some kind of weighted sum alignment, so wasting time to farm alignment doesnt pay off.