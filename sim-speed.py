import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

class BBTrajectory:
    def __init__(self, mass, diameter, initial_velocity, spin_rate, drag_coefficient=0.47, spin_decay_rate=0.0):
        self.mass = mass / 1000.0  # Convert to kg
        self.diameter = diameter / 1000.0  # Convert to meters
        self.area = np.pi * (self.diameter/2)**2
        self.drag_coefficient = drag_coefficient
        self.initial_velocity = np.array(initial_velocity)
        self.spin_rate = spin_rate * (2 * np.pi / 60)  # Convert RPM to rad/s
        self.magnus_coefficient = 0.1  # Added as class property
        self.g = 9.81  # Gravity acceleration
        self.air_density = 1.225  # kg/m³
        self.spin_decay_rate = spin_decay_rate  # Spin decay rate in rpm/s

    def calculate_drag_force(self, velocity):
        velocity_magnitude = np.linalg.norm(velocity)
        drag_magnitude = 0.5 * self.air_density * self.drag_coefficient * self.area * velocity_magnitude**2
        return -drag_magnitude * velocity / velocity_magnitude if velocity_magnitude > 0 else np.zeros(3)

    def calculate_magnus_force(self, velocity):
        velocity_magnitude = np.linalg.norm(velocity)
        if (velocity_magnitude == 0):
            return np.zeros(3)
        
        # Enhanced Magnus effect calculation
        spin_angular_velocity = self.spin_rate  # rad/s
        
        # Calculate lift using enhanced coefficient
        lift_coefficient = self.magnus_coefficient * (spin_angular_velocity * self.diameter) / (2 * velocity_magnitude)
        
        # Calculate lift force magnitude
        lift_force_magnitude = 0.5 * self.air_density * velocity_magnitude**2 * self.area * lift_coefficient
        
        # Direction of lift force (perpendicular to both velocity and spin axis)
        velocity_normalized = velocity / velocity_magnitude
        spin_axis = np.array([0, 1, 0])  # Y-axis rotation for backspin
        lift_direction = np.cross(velocity_normalized, spin_axis)
        
        if np.linalg.norm(lift_direction) > 0:
            lift_direction = lift_direction / np.linalg.norm(lift_direction)
        
        # Final Magnus force
        magnus_force = lift_force_magnitude * lift_direction  # Removed amplification
        
        return magnus_force

    def simulate_trajectory(self, dt=0.001, max_time=None):
        # Calculate a reasonable max_time based on initial conditions if not provided
        if max_time is None:
            # Estimate time using simple projectile motion as upper bound
            v0_y = 0  # Initial vertical velocity
            h0 = 1.5  # Initial height
            # Time to hit ground in vacuum (overestimate): t = (-v0 + sqrt(v0^2 + 2gh0))/g
            max_time = (-v0_y + np.sqrt(v0_y**2 + 2*self.g*h0))/self.g
            # Add 50% margin and account for horizontal distance
            max_time = max_time * 10 * (1 + np.linalg.norm(self.initial_velocity)/50)
        
        times = np.arange(0, max_time, dt)
        positions = np.zeros((len(times), 3))
        velocities = np.zeros((len(times), 3))
        
        # Initial conditions
        positions[0] = np.array([0, 0, 1.5])
        velocities[0] = self.initial_velocity
        
        for i in range(1, len(times)):
            # Apply spin decay as %/sec
            decay_fraction = self.spin_decay_rate / 100.0  # Convert % to fraction
            self.spin_rate = max(self.spin_rate * (1 - decay_fraction * dt), 0)
            
            # Calculate forces
            gravity_force = np.array([0, 0, -self.g * self.mass])
            drag_force = self.calculate_drag_force(velocities[i-1])
            magnus_force = self.calculate_magnus_force(velocities[i-1])
            
            # Sum forces and calculate acceleration
            total_force = gravity_force + drag_force + magnus_force
            acceleration = total_force / self.mass
            
            # Update velocity and position
            velocities[i] = velocities[i-1] + acceleration * dt
            positions[i] = positions[i-1] + velocities[i-1] * dt
            
            # Stop if BB hits ground and interpolate final position
            if positions[i, 2] < 0:
                # Interpolate to find exact ground intersection
                t_ground = times[i-1] - positions[i-1, 2] * dt / (positions[i, 2] - positions[i-1, 2])
                positions[i, :] = positions[i-1, :] + velocities[i-1, :] * (t_ground - times[i-1])
                positions[i, 2] = 0  # Set exact ground height
                return times[:i+1], positions[:i+1]
        
        return times, positions

def plot_trajectories(trajectories, labels):
    plt.figure(figsize=(12, 6))
    for traj, label in zip(trajectories, labels):
        times, positions = traj
        plt.plot(positions[:, 0], positions[:, 2], label=label)
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('BB Trajectories')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

def plot_interactive_trajectories(bb_diameter):
    # Set dark mode style
    plt.style.use('dark_background')
    
    # Create figure with space for sliders
    fig = plt.figure(figsize=(12, 8), facecolor='#1C1C1C')
    plot_ax = plt.axes([0.1, 0.3, 0.8, 0.4], facecolor='#2F2F2F')
    
    # Reserve space for colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.7])
    
    # Add a new axes for speed over distance
    speed_ax = fig.add_axes([0.1, 0.75, 0.2, 0.2], facecolor='#2F2F2F', sharex=plot_ax)
    
    # Available BB weights in grams
    bb_weights = [0.20, 0.23, 0.25, 0.28, 0.30, 0.32, 0.36, 0.40, 0.43, 0.46]
    
    # Energy slider instead of velocity slider (typical airsoft energies: 0.5J to 3J)
    energy_slider_ax = plt.axes([0.1, 0.05, 0.3, 0.03])
    energy_slider = Slider(energy_slider_ax, 'Energy (Joules)', 0.1, 3.0, 
                         valinit=1.0, valstep=0.1)
    
    spin_slider_ax = plt.axes([0.1, 0.1, 0.3, 0.03])
    weight_slider_ax = plt.axes([0.1, 0.15, 0.3, 0.03])
    
    spin_slider = Slider(spin_slider_ax, 'Spin Rate (rpm)', 0, 200000, valinit=120000, valstep=1000)
    weight_slider = Slider(weight_slider_ax, 'BB Weight (g)', 0, len(bb_weights)-1, 
                         valinit=2, valstep=1, valfmt='%1.2fg')
    
    def weight_format(val):
        return f'{bb_weights[int(val)]}g'
    weight_slider.valtext.set_text(weight_format(2))
    
    # Add target slider
    target_slider_ax = plt.axes([0.5, 0.05, 0.3, 0.03])  # Position next to velocity slider
    target_slider = Slider(target_slider_ax, 'Target Distance (m)', 1, 100, 
                         valinit=30, valstep=1)
    
    # Add spin decay slider
    spin_decay_slider_ax = plt.axes([0.1, 0.2, 0.3, 0.03])
    spin_decay_slider = Slider(spin_decay_slider_ax, 'Spin Decay (%/sec)', 0, 10, 
                               valinit=2, valstep=0.1)
    
    # Add TextBoxes for sliders
    # Energy TextBox
    energy_textbox_ax = plt.axes([0.42, 0.05, 0.1, 0.03])
    energy_textbox = TextBox(energy_textbox_ax, 'Energy (J)', initial=str(energy_slider.val), 
                             color='#2F2F2F', hovercolor='#555555')
    energy_textbox.text_disp.set_color('white')  # Set input text color
    
    # Spin Rate TextBox
    spin_textbox_ax = plt.axes([0.42, 0.1, 0.1, 0.03])
    spin_textbox = TextBox(spin_textbox_ax, 'Spin (rpm)', initial=str(spin_slider.val), 
                           color='#2F2F2F', hovercolor='#555555')
    spin_textbox.text_disp.set_color('white')  # Set input text color
    
    # BB Weight TextBox
    weight_textbox_ax = plt.axes([0.42, 0.15, 0.1, 0.03])
    weight_textbox = TextBox(weight_textbox_ax, 'Weight (g)', initial=weight_format(weight_slider.val), 
                             color='#2F2F2F', hovercolor='#555555')
    weight_textbox.text_disp.set_color('white')  # Set input text color
    
    # Target Distance TextBox
    target_textbox_ax = plt.axes([0.82, 0.05, 0.1, 0.03])
    target_textbox = TextBox(target_textbox_ax, 'Target (m)', initial=str(target_slider.val), 
                             color='#2F2F2F', hovercolor='#555555')
    target_textbox.text_disp.set_color('white')  # Set input text color
    
    # Spin Decay TextBox
    spin_decay_textbox_ax = plt.axes([0.42, 0.2, 0.1, 0.03])
    spin_decay_textbox = TextBox(spin_decay_textbox_ax, 'Spin Decay (%/s)', initial=str(spin_decay_slider.val), 
                                 color='#2F2F2F', hovercolor='#555555')
    spin_decay_textbox.text_disp.set_color('white')  # Set input text color
    
    # Define callback functions to synchronize TextBoxes with Sliders
    def submit_energy(text):
        try:
            val = float(text)
            energy_slider.set_val(val)
        except ValueError:
            pass  # Ignore invalid input
    
    def submit_spin(text):
        try:
            val = float(text)
            spin_slider.set_val(val)
        except ValueError:
            pass
    
    def submit_weight(text):
        try:
            val = float(text)
            # Find the closest weight index
            closest_weight = min(bb_weights, key=lambda x: abs(x - val))
            idx = bb_weights.index(closest_weight)
            weight_slider.set_val(idx)
        except ValueError:
            pass  # Ignore invalid input
    
    def submit_target(text):
        try:
            val = float(text)
            target_slider.set_val(val)
        except ValueError:
            pass
    
    def submit_spin_decay(text):
        try:
            val = float(text)
            spin_decay_slider.set_val(val)
        except ValueError:
            pass
    
    energy_textbox.on_submit(submit_energy)
    spin_textbox.on_submit(submit_spin)
    weight_textbox.on_submit(submit_weight)
    target_textbox.on_submit(submit_target)
    spin_decay_textbox.on_submit(submit_spin_decay)
    
    # Update TextBoxes when Sliders are changed
    def update_textboxes(val):
        energy_textbox.set_val(f"{energy_slider.val:.1f}")
        spin_textbox.set_val(f"{spin_slider.val:.0f}")
        weight_textbox.set_val(weight_format(weight_slider.val))
        target_textbox.set_val(f"{target_slider.val:.0f}")
        spin_decay_textbox.set_val(f"{spin_decay_slider.val:.1f}")
    
    # Register the update_textboxes function to all slider events
    energy_slider.on_changed(update_textboxes)
    spin_slider.on_changed(update_textboxes)
    weight_slider.on_changed(update_textboxes)
    target_slider.on_changed(update_textboxes)
    spin_decay_slider.on_changed(update_textboxes)
    
    def format_coord(x, y):
        # Find closest point on trajectory near the cursor, above or below
        if not hasattr(format_coord, 'positions') or not hasattr(format_coord, 'velocities'):
            return "No data"
        
        # Define tolerances
        x_tolerance = 0.5  # meters
        y_tolerance = 0.5  # meters
        
        # Filter points within x and y tolerance
        mask = (abs(format_coord.positions[:, 0] - x) < x_tolerance) & \
               (abs(format_coord.positions[:, 2] - y) < y_tolerance)
        
        if not np.any(mask):
            return "No trajectory data near cursor"
            
        # Get valid positions and find closest point
        valid_positions = format_coord.positions[mask]
        valid_velocities = format_coord.velocities[mask]
        
        distances = np.sqrt((valid_positions[:, 0] - x)**2 + 
                            (valid_positions[:, 2] - y)**2)
        idx = np.argmin(distances)
        
        if idx < len(valid_velocities):
            vel_ms = valid_velocities[idx]
            vel_fps = vel_ms / 0.3048
            pos_x = valid_positions[idx, 0]
            pos_z = valid_positions[idx, 2]
            return f'Distance: {pos_x:.1f}m, Height: {pos_z:.1f}m\nVelocity: {vel_fps:.0f} FPS ({vel_ms:.1f} m/s)'
        return "Out of range"
    
    def update(val):
        plot_ax.clear()
        speed_ax.clear()
        cbar_ax.clear()
        
        # Get values from sliders
        energy_joules = energy_slider.val
        weight = bb_weights[int(weight_slider.val)]
        mass_kg = weight / 1000.0  # Convert g to kg
        spin_decay_rate = spin_decay_slider.val
        
        # Calculate velocity from energy: E = 1/2 * m * v^2
        velocity_ms = np.sqrt((2 * energy_joules) / mass_kg)
        velocity_fps = velocity_ms / 0.3048  # Convert m/s to FPS
        
        spin_rate = spin_slider.val
        target_distance = target_slider.val
        
        bb = BBTrajectory(
            mass=weight,
            diameter=bb_diameter,
            initial_velocity=[velocity_ms, 0, 0],  # Use converted velocity
            spin_rate=spin_rate,
            spin_decay_rate=spin_decay_rate
        )
        
        times, positions = bb.simulate_trajectory()
        
        # Calculate velocities at each point
        velocities = np.zeros(len(positions))
        for i in range(len(positions)-1):
            dp = positions[i+1] - positions[i]
            dt = times[i+1] - times[i]
            velocities[i] = np.linalg.norm(dp/dt)
        velocities[-1] = velocities[-2]
        
        # Store positions and velocities for hover function
        format_coord.positions = positions
        format_coord.velocities = velocities
        
        # Draw ground line and target line
        max_x = max(np.max(positions[:, 0]) * 1.1, target_distance * 1.1)
        plot_ax.axhline(y=0, color='white', linestyle='-', alpha=0.3, zorder=0)
        plot_ax.axvline(x=target_distance, color='red', linestyle='--', alpha=0.5, label='Target')
        
        # Find time to target
        target_idx = np.argmin(np.abs(positions[:, 0] - target_distance))
        time_to_target = times[target_idx]
        target_height = positions[target_idx, 2]
        
        # Add target marker
        plot_ax.scatter(target_distance, target_height, 
                       color='yellow', s=100, marker='*', 
                       label=f'Time to target: {time_to_target:.3f}s\nHeight at target: {target_height:.2f}m')
        
        # Create color-mapped line
        points = plot_ax.scatter(positions[:, 0], positions[:, 2], 
                               c=velocities, cmap='plasma',  # Changed colormap
                               s=1, label=f'{weight}g BB')
        
        # Add impact point marker with distance in meters and feet
        plot_ax.scatter(positions[-1, 0], positions[-1, 2], 
                       color='red', s=50, marker='x', 
                       label=f'Impact: {positions[-1, 0]:.1f}m ({positions[-1, 0]*3.28084:.1f}ft)')
        
        # Set axis limits and style
        plot_ax.set_xlim(0, max_x)
        speed_ax.set_xlim(0, max_x)
        plot_ax.set_ylim(-0.1, np.max(positions[:, 2]) * 1.1)
        plot_ax.set_facecolor('#2F2F2F')
        
        # Add colorbar and labels with dark theme
        cbar = plt.colorbar(points, cax=cbar_ax)
        cbar_ax.set_ylabel('Velocity (m/s)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        plot_ax.set_xlabel('Distance (m)', color='white')
        plot_ax.set_ylabel('Height (m)', color='white')
        plot_ax.set_title(f'BB Trajectory (Energy: {energy_joules:.1f}J, Velocity: {velocity_fps:.0f} FPS = {velocity_ms:.1f} m/s)', 
                         color='white', pad=10)
        plot_ax.grid(True, alpha=0.2)
        plot_ax.legend(facecolor='#2F2F2F', edgecolor='gray')
        
        # Add coordinate formatter for hover
        plot_ax.format_coord = format_coord
        
        # Update tick colors
        plot_ax.tick_params(colors='white')
        
        # Calculate cumulative distance traveled
        cumulative_distance = np.zeros(len(positions))
        cumulative_distance[1:] = np.cumsum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))
        
        # Calculate speeds
        speeds = velocities  # Already calculated speeds
        
        # Plot speed over traveled distance
        speed_ax.plot(cumulative_distance, speeds, color='cyan')
        speed_ax.set_xlabel('Traveled Distance (m)', color='white')
        speed_ax.set_ylabel('Speed (m/s)', color='white')
        speed_ax.set_title('Speed over Traveled Distance', color='white')
        speed_ax.grid(True, alpha=0.2)
        speed_ax.tick_params(colors='white')
        speed_ax.set_facecolor('#2F2F2F')
        
        weight_slider.valtext.set_text(weight_format(weight_slider.val))
        fig.canvas.draw_idle()
    
    energy_slider.on_changed(update)
    spin_slider.on_changed(update)
    weight_slider.on_changed(update)
    target_slider.on_changed(update)
    spin_decay_slider.on_changed(update)
    
    update(None)  # Initial plot
    plt.show()

def main():
    bb_diameter = 6  # 6mm BB
    plot_interactive_trajectories(bb_diameter)

if __name__ == "__main__":
    main()
