import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects
from termcolor import colored, cprint

# Data Logger object
class dataLogger(object):
   """docstring for dataLogger"""
    def __init__(self):
        self.ticktime = []
        self.ticks = []  # time
        self.x_ref = []  # aimed pos
        self.y_ref = []
        self.z_ref = []
        self.x_pos = []  # actuall pos
        self.y_pos = []
        self.z_pos = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.u_smc = []
        self.u_palm = []
        self.u_c = []
        self.nNodes = []
        self.timestr = time.strftime("%Y%m%d-%H%M%S")

    def appendStateData(self, t, xr, yr, zr, xa, ya, za, vx, vy, vz):
        self.ticktime.append(t)
        self.x_ref.append(xr)
        self.y_ref.append(yr)
        self.z_ref.append(zr)
        self.x_pos.append(xa)
        self.y_pos.append(ya)
        self.z_pos.append(za)
        self.vx.append(vx)
        self.vy.append(vy)
        self.vz.append(vz)

    def appendControlData(self, t, usmc, upalm, u, node):
        self.ticks.append(t)
        self.u_smc.append(usmc)
        self.u_palm.append(upalm)
        self.u_c.append(u)
        self.nNodes.append(node)

    def saveAll(self):
        text = colored('Saving all logged data..', 'green', attrs=['reverse', 'blink'])
        print(text)
        with open("FlightLog" + self.timestr + ".txt", "w") as output:
            output.writelines(map("{};{};{};{};{};{};{};{};{};{}\n".format, self.ticktime,
                                  self.x_ref, self.y_ref, self.z_ref,
                                  self.x_pos, self.y_pos, self.z_pos,
                                  self.vx, self.vy, self.vz))

    def plotFigure(self):
        print('Plotting the results..')
        exr = np.array(self.x_ref)
        ex = np.array(self.x_pos)
        eyr = np.array(self.y_ref)
        ey = np.array(self.y_pos)
        ezr = np.array(self.z_ref)
        ez = np.array(self.z_pos)
        sqrex = np.multiply((exr - ex), (exr - ex))
        sqrey = np.multiply((eyr - ey), (eyr - ey))
        sqrez = np.multiply((ezr - ez), (ezr - ez))
        rmsex = np.sqrt(sum(sqrex) / np.size(ex))
        rmsey = np.sqrt(sum(sqrey) / np.size(ey))
        rmsez = np.sqrt(sum(sqrez) / np.size(ez))
        print('RMSE XYZ = ', rmsex, rmsey, rmsez)

        # calculate velocity
        time_np = np.array(self.ticktime)
        x_np = np.array(self.x_pos)
        y_np = np.array(self.y_pos)
        z_np = np.array(self.z_pos)
        time_steps = (time_np[1:] - time_np[:-1])
        ref_vx = (x_np[1:] - x_np[:-1]) / time_steps
        ref_vy = (y_np[1:] - y_np[:-1]) / time_steps
        ref_vz = (z_np[1:] - z_np[:-1]) / time_steps
        tot_v = (ref_vx ** 2 + ref_vy ** 2 + ref_vz ** 2) ** 0.5
        print("v_max = ", max(tot_v))

        fig = plt.figure("XYZ plot", figsize=(16, 9))
        ax1 = plt.subplot2grid((6, 8), (0, 0), projection='3d', colspan=4, rowspan=6)
        ax1.plot3D(self.x_ref, self.y_ref, self.z_ref, 'k--', label='Target', color='blue')
        cmap = plt.get_cmap('autumn')
        ax1.scatter(self.x_pos[1:], self.y_pos[1:], self.z_pos[1:], c=cmap(1 - np.abs(tot_v) / abs(max(tot_v))),
                    edgecolor='none', label='Drone trajectory')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_xlim3d(-2, 2)
        ax1.set_ylim3d(-2, 2)
        ax1.set_zlim3d(0, 4)
        ax1.legend()

        ax2 = plt.subplot2grid((6, 8), (0, 4), colspan=2, rowspan=2)
        ax2.plot(self.ticktime, self.x_ref, self.ticktime, self.x_pos)
        ax2.grid(True)
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('X (m)')

        ax3 = plt.subplot2grid((6, 8), (2, 4), colspan=2, rowspan=2)
        ax3.plot(self.ticktime, self.y_ref, self.ticktime, self.y_pos)
        ax3.grid(True)
        ax3.set_xlabel('time (s)')
        ax3.set_ylabel('Y (m)')

        ax4 = plt.subplot2grid((6, 8), (4, 4), colspan=2, rowspan=2)
        ax4.plot(self.ticktime, self.z_ref, self.ticktime, self.z_pos)
        ax4.grid(True)
        ax4.set_xlabel("time(s)")
        ax4.set_ylabel('Z (m)')
        ax4.text(50, 0.5, 'RMSE Z = %.2f' % rmsez, fontsize=10)

        # calculate velocety
        time_np = np.array(self.ticktime)
        x_np = np.array(self.x_pos)
        y_np = np.array(self.y_pos)
        z_np = np.array(self.z_pos)
        time_steps = (time_np[1:] - time_np[:-1])
        ref_vx = (x_np[1:] - x_np[:-1]) / time_steps
        ref_vy = (y_np[1:] - y_np[:-1]) / time_steps
        ref_vz = (z_np[1:] - z_np[:-1]) / time_steps

        ax2 = plt.subplot2grid((6, 8), (0, 6), colspan=2, rowspan=2)
        ax2.plot(self.ticktime, self.vx, 'm*')
        ax2.plot(self.ticktime[1:], ref_vx, 'b^')
        ax2.grid(True)
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('Vx (m)')

        ax3 = plt.subplot2grid((6, 8), (2, 6), colspan=2, rowspan=2)
        ax3.plot(self.ticktime, self.vy, 'm*')
        ax3.plot(self.ticktime[1:], ref_vy, 'b^')
        ax3.grid(True)
        ax3.set_xlabel('time (s)')
        ax3.set_ylabel('Vy (m)')

        ax4 = plt.subplot2grid((6, 8), (4, 6), colspan=2, rowspan=2)
        ax4.plot(self.ticktime, self.vz, 'm*')
        ax4.plot(self.ticktime[1:], ref_vz, 'b^')
        ax4.grid(True)
        ax4.set_xlabel("time(s)")
        ax4.set_ylabel('Vz (m)')
        ax4.text(50, 0.5, 'RMSE Z = %.2f' % rmsez, fontsize=10)

        fig.tight_layout()
        fig.subplots_adjust(hspace=1)

        plt.show()

    def plotControlData(self):
        print('Plotting the control signal..')
        plt.figure("Control signal")
        plt.plot(self.ticks, self.u_smc, label='SMC')
        plt.plot(self.ticks, self.u_palm, label='NN')
        plt.grid(True)
        plt.xlabel("time(s)")
        plt.ylabel('control signal')
        plt.legend()

        plt.show()