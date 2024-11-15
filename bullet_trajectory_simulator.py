import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Physical constants
AIR_DENSITY = 1.225  # kg/m³
GRAVITY = 9.81  # m/s²
DRAG_COEFFICIENT = 0.47  # Sphere
MAGNUS_COEFFICIENT = 0.1  # Approximate for BBs
BARREL_LENGTH = 0.285  # m (285mm)
BARREL_DIAMETER = 0.0061  # m (6.1mm)
FPS_TO_MS = 0.3048  # 1 ft/s = 0.3048 m/s

class BBTrajectory:
    def __init__(self, mass_g, diameter_mm, initial_velocity_ms, backspin_rps):
        # Convert units to SI
        self.mass = mass_g / 1000  # kg
        self.diameter = diameter_mm / 1000  # m
        self.velocity = initial_velocity_ms  # m/s
        self.spin_rate = backspin_rps * 2 * np.pi  # rad/s
        
        self.area = np.pi * (self.diameter/2)**2
        self.dt = 0.001  # Time step for simulation
        self.clearance = (BARREL_DIAMETER - self.diameter) / 2  # Barrel-to-BB clearance
        
    def calculate_forces(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return np.zeros(3)
            
        # Drag force
        drag_force = -0.5 * AIR_DENSITY * self.area * DRAG_COEFFICIENT * speed * velocity
        
        # Magnus force (perpendicular to velocity and spin axis)
        spin_vector = np.array([0, 0, self.spin_rate])  # Backspin around z-axis
        magnus_force = (MAGNUS_COEFFICIENT * self.area * AIR_DENSITY * 
                       np.cross(spin_vector, velocity))
        
        # Total force including gravity
        total_force = drag_force + magnus_force + np.array([0, -self.mass * GRAVITY, 0])
        
        return total_force
        
    def simulate(self, max_time=10.0):  # Increased from 5.0 to 10.0 seconds for higher spin rates
        positions = []
        times = []
        last_height = 1.5  # Track height for better interpolation
        
        # Adjust starting position for barrel length
        initial_x = BARREL_LENGTH * np.cos(0)  # Assuming horizontal barrel
        initial_y = 1.5 + BARREL_LENGTH * np.sin(0)  # Height + barrel offset
        pos = np.array([initial_x, initial_y, 0.])
        vel = np.array([self.velocity, 0., 0.])
        t = 0
        
        while (pos[1] >= 0 or last_height > 0) and t < max_time:  # Added last_height check
            last_height = pos[1]
            positions.append(pos.copy())
            times.append(t)
            
            force = self.calculate_forces(vel)
            acc = force / self.mass
            
            # Euler integration with smaller time step for high spin rates
            self.dt = 0.0005 if self.spin_rate > 12 * np.pi else 0.001  # Adjust time step for high spin
            vel += acc * self.dt
            pos += vel * self.dt
            t += self.dt
        
        # More precise ground impact interpolation
        impact_distance = 0
        if len(positions) > 1 and positions[-1][1] < 0:
            prev_pos = positions[-2]
            curr_pos = positions[-1]
            t_frac = -prev_pos[1] / (curr_pos[1] - prev_pos[1])
            x_impact = prev_pos[0] + (curr_pos[0] - prev_pos[0]) * t_frac
            positions[-1] = np.array([x_impact, 0.0, 0.0])
            impact_distance = x_impact
            
        return np.array(positions), np.array(times), impact_distance

def plot_trajectories(bbs):
    plt.figure(figsize=(12, 6))
    
    for bb, label in bbs:
        positions, times = bb.simulate()
        plt.plot(positions[:, 0], positions[:, 1], label=label)
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('Airsoft BB Trajectories')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

def interactive_plot(bb_weights, diameter):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.2, right=0.95, left=0.08)  # Increased right margin
    
    # Create slider axes and widgets
    ax_spin = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]
    ax_fps = plt.axes([0.2, 0.05, 0.6, 0.03])   # Second slider below first
    
    spin_slider = Slider(
        ax=ax_spin,
        label='Rotations/sec',
        valmin=0,
        valmax=20,
        valinit=2,
        valstep=0.5
    )
    
    fps_slider = Slider(
        ax=ax_fps,
        label='Muzzle Velocity (FPS)',
        valmin=200,
        valmax=500,
        valinit=328,  # ~100 m/s
        valstep=5
    )
    
    def update(val):
        ax.clear()
        rps = spin_slider.val
        fps = fps_slider.val
        velocity_ms = fps * FPS_TO_MS
        
        max_range = 0
        max_height = 0
        # Create BBs with current slider values
        for weight in bb_weights:
            bb = BBTrajectory(weight, diameter, velocity_ms, rps)
            positions, _, impact = bb.simulate()
            line = ax.plot(positions[:, 0], positions[:, 1], 
                         label=f"{weight}g BB")[0]
            
            # Updated annotation to include both range and weight
            annotation_x = impact - 1
            annotation_y = 0.2
            ax.annotate(f'{weight}g - {impact:.1f}m', 
                       xy=(impact, 0), 
                       xytext=(annotation_x, annotation_y),
                       color=line.get_color(),
                       fontsize=10)
            
            max_range = max(max_range, impact)
            max_height = max(max_height, np.max(positions[:, 1]))
        
        ax.set_xlabel('Distance (m)', fontsize=12)
        ax.set_ylabel('Height (m)', fontsize=12)
        ax.set_title(f'Airsoft BB Trajectories (Spin: {rps} RPS, Velocity: {fps} FPS)', fontsize=14)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        # Set axis limits with padding
        ax.set_xlim(-1, max_range * 1.05)  # Reduced x padding
        ax.set_ylim(-0.2, max_height * 1.1)  # Reduced y padding
        
        # Remove aspect ratio constraint
        ax.set_aspect('auto')
        
        # Make tick labels larger
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        fig.canvas.draw_idle()
    
    # Set initial plot
    update(100)
    
    # Register the update function with both sliders
    spin_slider.on_changed(update)
    fps_slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    bb_weights = [0.20, 0.25, 0.30, 0.40]
    interactive_plot(bb_weights, 6)  # 6mm diameter
