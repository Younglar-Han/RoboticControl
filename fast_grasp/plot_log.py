import matplotlib.pyplot as plt

# Initialize a dictionary to store positions for each joint
joint_positions = {}

# Open the log file and read its contents
with open('joint_positions.log', 'r') as file:
    for line in file:
        # Parse the line to extract the joint name and position
        joint, position = line.split(': ')
        position = float(position)

        # Add the position to the list for this joint
        if joint in joint_positions:
            joint_positions[joint].append(position)
        else:
            joint_positions[joint] = [position]

# Create a plot for each joint
# for joint, positions in joint_positions.items():
#     plt.figure()
#     plt.plot(positions)
#     plt.title(f'Positions for {joint}')
#     plt.xlabel('Time step')
#     plt.ylabel('Position')
#     plt.show()
    
# plot the joint positions in one figure
plt.figure()
for joint, positions in joint_positions.items():
    plt.plot(positions, label=joint)
plt.title('Joint Positions')
plt.xlabel('Time step')
plt.ylabel('Position')

plt.legend()
plt.show()