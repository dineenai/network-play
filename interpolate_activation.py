
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import os

files_dir = "/home/ainedineen/motion_dnns/sugarcubes/sarah_sim"
save_act_loc = os.path.join(files_dir, 'activation')

# load activation with appropriate colnames
# eg. header_list = ["Name", "Dept", "Start Date"]
#     df = pd.read_csv("sample_file.csv", names=header_list)

header_list = ["time", "activation"]
data = pd.read_csv(f'{files_dir}/activation3Dtimecourse_1pct', sep="   ", header=None, names=header_list)
print(data.head)
print(data['time'][7]) #row
print(data.activation[7])

TR = 2.76
half_TR = TR/2

from matplotlib import pyplot as plt
fig, ax = plt.subplots()
fig.suptitle(f'Activation over time')
ax.plot(data.time, data.activation)


print(files_dir)

save_act = os.path.join(save_act_loc, 'activation_timecourse.png')

plt.savefig(save_act)
print(f'Saved: {save_act}')



n_vols = 24
times = []

# n number of steps not upper range!
for i in range(n_vols):
    #  time of approx activation
    times.append(half_TR * i)

print(times)
print(len(times))



# use scipy to interpolate values
# linear interpolation is slight underestimate of convex wave!

x = data.time 
y = data.activation
f = interpolate.interp1d(x, y)

# list to array
new_x_times = np.array(times)

new_y_est_act = f(new_x_times)   # use interpolation function returned by `interp1d`
plt.plot(x, y, '-', new_x_times, new_y_est_act, 'o')

save_act_fig = os.path.join(save_act_loc, 'interpolated_activation_timecourse.png')


plt.savefig(save_act_fig)
print(f'Saved: {save_act_fig}')




df = pd.DataFrame({'mid_vol_time':new_x_times, 'est_activation':new_y_est_act})
print(df.head)
save_est_act = os.path.join(save_act_loc, 'est_act_per_vol.csv')
# remove index as additional index added when loading!
df.to_csv(save_est_act, index=False)

# test load
data = pd.read_csv(save_est_act)
print(data.head)

print(data.est_activation[0])
print(data.est_activation[7])
print(data.est_activation[23])




# to get average activation value (y)
files_dir = "/home/ainedineen/motion_dnns/sugarcubes/sarah_sim"
save_act_loc = os.path.join(files_dir, 'activation')
save_est_act = os.path.join(save_act_loc, 'est_act_per_vol.csv')

# test load
data = pd.read_csv(save_est_act)
vol_no = 7
print(data.est_activation[vol_no])