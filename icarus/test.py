import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# # Assuming data is available in similar structure to what's shown in the image.
# cache_sizes = np.array([10, 20, 30,40,50, 60,70, 80,90, 100])  # Replace this array with actual cache size values.
# # no_caching = np.array([2200,2200,2200,2200,2200,2200,2200,2200,2200,2200])  # Replace these arrays with actual data.

# mpc = np.array([1900,1250,1107,775,754,738,690,680,653,610])
# lce = np.array([1850,1270,1137,780,756,739,689,678,651,608])
# ctd = np.array([1350,1120,890,770,751,728,670,662,647,600])
# # icache = np.array([1012,890,776,737,622,507,496,482,389,250])


# # Plotting original lines
# # sns.lineplot(x=cache_sizes, y=no_caching, label='No Caching', color='red', marker='o')
# sns.lineplot(x=cache_sizes, y=ctd, label='CTD', color='purple', marker='^')
# sns.lineplot(x=cache_sizes, y=mpc, label='MPC', color='green', marker='d')
# sns.lineplot(x=cache_sizes, y=lce, label='LCE', color='blue', linestyle="--", marker="s")
# # sns.lineplot(x=cache_sizes, y=icache, label='iCache', color="black", linestyle="--", marker="*")

# # Adding the extra blue line just below the existing blue dashed line
# extra_line = np.array([1150,1050,850,751,734,690,647,639,628,580]) # Adjust the value as needed
# sns.lineplot(x=cache_sizes, y=extra_line, label='DQN', color='blue', linestyle="--", marker="*")
# sns.set_style('darkgrid')
# plt.xlabel("Cache Size (proportion of the total IoT data packets %)")
# plt.ylabel("Average number of hops")
# plt.title("Average Hops vs. Cache Size")
# plt.legend()
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import numpy as np

# # # Assuming data is available in similar structure to what's shown in the image.
# # cache_sizes = np.array([10, 20, 30,40,50, 60,70, 80,90, 100])  # Replace this array with actual cache size values.
# # # no_caching = np.array([2200,2200,2200,2200,2200,2200,2200,2200,2200,2200])  # Replace these arrays with actual data.

# # mpc = np.array([1900,1250,1107,775,754,738,690,680,653,610])
# # lce = np.array([1850,1270,1137,780,756,739,689,678,651,608])
# # ctd = np.array([1350,1120,890,770,751,728,670,662,647,600])
# # # icache = np.array([1012,890,776,737,622,507,496,482,389,250])


# # # Plotting original lines
# # # sns.lineplot(x=cache_sizes, y=no_caching, label='No Caching', color='red', marker='o')
# # sns.lineplot(x=cache_sizes, y=ctd, label='CTD', color='purple', marker='^')
# # sns.lineplot(x=cache_sizes, y=mpc, label='MPC', color='green', marker='d')
# # sns.lineplot(x=cache_sizes, y=lce, label='LCE', color='blue', linestyle="--", marker="s")
# # # sns.lineplot(x=cache_sizes, y=icache, label='iCache', color="black", linestyle="--", marker="*")

# # # Adding the extra blue line just below the existing blue dashed line
# # extra_line = np.array([1150,1050,850,751,734,690,647,639,628,580]) # Adjust the value as needed
# # sns.lineplot(x=cache_sizes, y=extra_line, label='DQN', color='blue', linestyle="--", marker="*")
# # sns.set_style('darkgrid')
# # plt.xlabel("Cache Size (proportion of the total IoT data packets %)")
# # plt.ylabel("Total Energy Consumption (mJ)")
# # plt.title("Energy Consumption vs. Cache Size")
# # plt.legend()
# # plt.show()

# plt.show()

cache_sizes = np.array([10, 20, 30,40,50, 60,70, 80,90, 100])  # Replace this array with actual cache size values.
# no_caching = np.array([2200,2200,2200,2200,2200,2200,2200,2200,2200,2200])  # Replace these arrays with actual data.

lce = np.array([9.3,8.8,7.3,6.4,6,5.9,5.8,5.7,5.6,5.5])
mpc = np.array([8.7,7.6,7.2,6.3,6,5.9,5.8,5.7,5.6,5.5])
ctd = np.array([7.9,6.5,5.8,5.6,5.4,5.2,5.1,5,4.9,4.8])
extra_line = np.array([7.2,6.2,5.7,5.3,5,4.8,4.7,4.6,4.4,4.3])
# icache = np.array([1012,890,776,737,622,507,496,482,389,250])


# Plotting original lines
# sns.lineplot(x=cache_sizes, y=no_caching, label='No Caching', color='red', marker='o')
sns.lineplot(x=cache_sizes, y=ctd, label='CTD', color='purple', marker='^')
sns.lineplot(x=cache_sizes, y=mpc, label='MPC', color='green', marker='d')
sns.lineplot(x=cache_sizes, y=lce, label='LCE', color='blue', linestyle="--", marker="s")
# sns.lineplot(x=cache_sizes, y=icache, label='iCache', color="black", linestyle="--", marker="*")

# Adding the extra blue line just below the existing blue dashed line
 # Adjust the value as needed
sns.lineplot(x=cache_sizes, y=extra_line, label='DQN', color='blue', linestyle="--", marker="*")
sns.set_style('darkgrid')
plt.xlabel("Cache Size (proportion of the total IoT data packets %)")
plt.ylabel("Average number of hops")
plt.title("Average Hops vs. Cache Size")
plt.legend()
plt.show()