import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

# Initializing constants

G               = 9.8       # gravitational acceleration (m/s^2)
L1              = 1.0       # rod length (m)
L2              = 1.6       # second rod length (m)
M1              = 3.0       # pendulum mass (kg)
M2              = 1.0       # second pendulum mass (kg)
THETA_INIT      = 180.0     # initial angle (degrees)
THETA_INIT2     = 180.0     # second initial angle (degrees)
W_INIT          = 0.0       # initial angular velocity (degrees/s)
W_INIT2         = 0.0       # second initial angular velocity (degrees/s)

DELTA_T         = 0.05      # time interval (s)
t = np.arange(0, 100, DELTA_T)

INIT_STATE = np.radians([THETA_INIT, W_INIT, THETA_INIT2, W_INIT2])

# Define methods for computing the angle and the coordinates

def func(alpha, t):
    
    sum_masses = M1 + M2
    diff_angles_sin = np.sin(alpha[0] - alpha[2])
    diff_angles_cos = np.cos(alpha[0] - alpha[2])

    first_der1 = alpha[1]
    first_der2 = alpha[3]
    second_der1 = (-L1*first_der1**2*diff_angles_sin + G*np.sin(alpha[2]) - 
                   (M2*L2*first_der2**2*diff_angles_sin + 
                    sum_masses*G*np.sin(alpha[0]))/(M2*diff_angles_cos)) * (M2*diff_angles_cos)/(sum_masses*L1 - M2*L1*diff_angles_cos**2)
    second_der2 = (L1*first_der1**2*diff_angles_sin - G*np.sin(alpha[2]) - L1*second_der1*diff_angles_cos) / L2
    
    return (first_der1, second_der1, first_der2, second_der2)

theta = integrate.odeint(func, INIT_STATE, t)

x1 = L1*np.sin(theta[:, 0])
y1 = -L1*np.cos(theta[:, 0])
data1  = np.array([x1, y1])

x2 = x1 + L2*np.sin(theta[:, 2])
y2 = y1 - L2*np.cos(theta[:, 2])
data2  = np.array([x2, y2])

# Creating the figure and plotting the data

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
line1, = ax.plot([], [], '-', color = "green", alpha = 0.3, lw=1)
line2, = ax.plot([], [], '-', color = "orange", alpha = 0.5, lw=1)
time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line, line1, line2, time_text
    
def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    line1.set_data(data1)
    line2.set_data(data2)
    time_text.set_text(time_template % (i*DELTA_T))
    return line, line1, line2, time_text

ani = animation.FuncAnimation(fig, animate, range(1, len(t)), interval=DELTA_T*1000, blit=True, init_func=init)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()