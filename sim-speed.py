import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import time
from functools import wraps

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
        C = 0.5 * self.air_density * self.drag_coefficient * self.area
        drag_magnitude = C * velocity_magnitude**2
        return -drag_magnitude * velocity / velocity_magnitude if velocity_magnitude > 0 else np.zeros(3)

    def calculate_magnus_force(self, velocity):
        velocity_magnitude = np.linalg.norm(velocity)
        if (velocity_magnitude == 0):
            return np.zeros(3)
        
        C = 0.5 * self.air_density * self.area
        spin_angular_velocity = self.spin_rate
        lift_coefficient = self.magnus_coefficient * (spin_angular_velocity * self.diameter) / (2 * velocity_magnitude)
        lift_force_magnitude = C * velocity_magnitude**2 * lift_coefficient
        
        velocity_normalized = velocity / velocity_magnitude
        spin_axis = np.array([0, 1, 0])  # Y-axis rotation for backspin
        lift_direction = np.cross(velocity_normalized, spin_axis)
        
        if np.linalg.norm(lift_direction) > 0:
            lift_direction = lift_direction / np.linalg.norm(lift_direction)
        
        return lift_force_magnitude * lift_direction

    def simulate_trajectory(self, dt=0.001, max_time=None):
        if max_time is None:
            v0_y = 0  # Initial vertical velocity
            h0 = 1.5  # Initial height
            max_time = (-v0_y + np.sqrt(v0_y**2 + 2*self.g*h0))/self.g
            max_time = max_time * 10 * (1 + np.linalg.norm(self.initial_velocity)/50)
        
        times = np.arange(0, max_time, dt)
        positions = np.zeros((len(times), 3))
        velocities = np.zeros((len(times), 3))
        
        positions[0] = np.array([0, 0, 1.5])
        velocities[0] = self.initial_velocity
        
        for i in range(1, len(times)):
            decay_fraction = self.spin_decay_rate / 100.0  # Convert % to fraction
            self.spin_rate = max(self.spin_rate * (1 - decay_fraction * dt), 0)
            
            gravity_force = np.array([0, 0, -self.g * self.mass])
            drag_force = self.calculate_drag_force(velocities[i-1])
            magnus_force = self.calculate_magnus_force(velocities[i-1])
            friction_force = -self.f * velocities[i-1] if hasattr(self, 'f') else np.zeros(3)
            
            total_force = gravity_force + drag_force + magnus_force + friction_force
            acceleration = total_force / self.mass
            
            velocities[i] = velocities[i-1] + acceleration * dt
            positions[i] = positions[i-1] + velocities[i-1] * dt
            
            if positions[i, 2] < 0:
                t_ground = times[i-1] - positions[i-1, 2] * dt / (positions[i, 2] - positions[i-1, 2])
                positions[i, :] = positions[i-1, :] + velocities[i-1, :] * (t_ground - times[i-1])
                positions[i, 2] = 0
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
    # Setup figure and axes
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor='#1C1C1C')
    status_text = fig.text(0.92, 0.95, '', color='white', fontsize=10)
    
    # Create axes
    plot_ax = plt.axes([0.1, 0.25, 0.8, 0.5], facecolor='#2F2F2F')
    speed_ax = fig.add_axes([0.1, 0.8, 0.4, 0.15], facecolor='#2F2F2F')
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    
    # Add section titles
    fig.text(0.1, 0.18, 'BB Properties', color='white', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.18, 'Target Settings', color='white', fontsize=12, fontweight='bold')
    fig.text(0.1, 0.08, 'Spin Settings', color='white', fontsize=12, fontweight='bold')
    
    # Slider setup helpers
    def create_slider(ax, label, vmin, vmax, vinit, **kwargs):
        slider_style = {
            'color': '#4A90E2',
            'initcolor': 'none',
            'track_color': '#404040'
        }
        slider_style.update(kwargs)
        return Slider(ax, label, vmin, vmax, valinit=vinit, **slider_style)

    def create_textbox(ax, initial):
        tb = TextBox(ax, '', initial=str(initial), color='#2F2F2F', hovercolor='#404040')
        tb.text_disp.set_color('white')
        return tb

    # Create sliders and textboxes
    bb_weights = [0.20, 0.23, 0.25, 0.28, 0.30, 0.32, 0.36, 0.40, 0.43, 0.46]
    
    # Create all sliders with their axes
    sliders = {
        'energy': create_slider(plt.axes([0.1, 0.15, 0.25, 0.02]), 'Energy (J)', 0.1, 3.0, 1.0, valstep=0.1),
        'weight': create_slider(plt.axes([0.1, 0.12, 0.25, 0.02]), 'Weight (g)', 0, len(bb_weights)-1, 3, valstep=1),
        'target': create_slider(plt.axes([0.5, 0.15, 0.25, 0.02]), 'Target (m)', 1, 100, 30),
        'spin': create_slider(plt.axes([0.1, 0.05, 0.25, 0.02]), 'Spin (rpm)', 0, 200000, 120000, valstep=1000),
        'spin_decay': create_slider(plt.axes([0.1, 0.02, 0.25, 0.02]), 'Decay (%/s)', 0, 10, 3.3, valstep=0.1),
        'tolerance': create_slider(plt.axes([0.5, 0.12, 0.25, 0.02]), 'BB Tolerance (mm)', 0.001, 0.05, 0.005, valstep=0.001),
        'weight_tolerance': create_slider(plt.axes([0.5, 0.09, 0.25, 0.02]), 'Weight Tolerance (%)', 0.1, 5.0, 1.0, valstep=0.1)
    }
    
    # Create corresponding textboxes
    textboxes = {
        'energy': create_textbox(plt.axes([0.37, 0.15, 0.08, 0.02]), sliders['energy'].val),
        'weight': create_textbox(plt.axes([0.37, 0.12, 0.08, 0.02]), f"{bb_weights[int(sliders['weight'].val)]}g"),
        'target': create_textbox(plt.axes([0.77, 0.15, 0.08, 0.02]), sliders['target'].val),
        'spin': create_textbox(plt.axes([0.37, 0.05, 0.08, 0.02]), sliders['spin'].val),
        'spin_decay': create_textbox(plt.axes([0.37, 0.02, 0.08, 0.02]), sliders['spin_decay'].val),
        'tolerance': create_textbox(plt.axes([0.77, 0.12, 0.08, 0.02]), sliders['tolerance'].val),
        'weight_tolerance': create_textbox(plt.axes([0.77, 0.09, 0.08, 0.02]), sliders['weight_tolerance'].val)
    }

    # Helper functions for format and update
    def weight_format(val):
        return f'{bb_weights[int(val)]}g'

    def format_coord(x, y):
        if not hasattr(format_coord, 'positions') or not hasattr(format_coord, 'velocities'):
            return "No data"
        
        # Find data at current x position
        x_tolerance = 0.5  # meters
        
        # Find closest x position in trajectory data
        distances_x = np.abs(format_coord.positions[:, 0] - x)
        if np.min(distances_x) > x_tolerance:
            return "No trajectory data at this distance"
            
        idx = np.argmin(distances_x)
        pos_x = format_coord.positions[idx, 0]
        pos_z = format_coord.positions[idx, 2]
        vel_ms = format_coord.velocities[idx]
        vel_fps = vel_ms / 0.3048
        
        # Get tolerance info at this x position
        if hasattr(format_coord, 'min_traj') and hasattr(format_coord, 'max_traj') and hasattr(format_coord, 'x_interp'):
            min_height = np.interp(x, format_coord.x_interp, format_coord.min_traj)
            max_height = np.interp(x, format_coord.x_interp, format_coord.max_traj)
            tolerance_info = f'Spread at {pos_x:.1f}m: {min_height:.2f}m - {max_height:.2f}m'
        else:
            tolerance_info = ''
            
        # Calculate energies at this point
        mass = format_coord.mass  # kg
        kinetic_energy = 0.5 * mass * vel_ms**2  # Joules
        potential_energy = mass * 9.81 * pos_z    # Joules
        total_energy = kinetic_energy + potential_energy
        
        return (f'Distance: {pos_x:.1f}m, Height: {pos_z:.1f}m '
                f'{tolerance_info}\n'
                f'Velocity: {vel_fps:.0f} FPS ({vel_ms:.1f} m/s) '
                f'Energy: {total_energy:.2f}J)'
				)

    def debounce(wait):
        def decorator(fn):
            last_called = [0]
            timer = [None]
            is_calculating = [False]
            
            @wraps(fn)
            def debounced(*args, **kwargs):
                def call_function():
                    if is_calculating[0]:
                        return
                    is_calculating[0] = True
                    status_text.set_text('Calculating...')
                    status_text.set_color('yellow')
                    fig.canvas.draw_idle()
                    
                    try:
                        fn(*args, **kwargs)
                    finally:
                        is_calculating[0] = False
                        status_text.set_text('Ready')
                        status_text.set_color('lime')
                        fig.canvas.draw_idle()
                
                last_called[0] = time.time()
                
                if timer[0]:
                    timer[0].stop()
                    timer[0] = None
                
                timer[0] = fig.canvas.new_timer(interval=wait*1000)
                timer[0].add_callback(call_function)
                timer[0].start()
                
                # Show calculating immediately when input changes
                if not is_calculating[0]:
                    status_text.set_text('Calculating...')
                    status_text.set_color('yellow')
                    fig.canvas.draw_idle()
            
            return debounced
        return decorator

    def calculate_trajectory_with_diameter(diameter, *args, **kwargs):
        bb = BBTrajectory(diameter=diameter, *args, **kwargs)
        return bb.simulate_trajectory()

    @debounce(1.0)
    def update(val):
        plot_ax.clear()
        speed_ax.clear()
        cbar_ax.clear()
        
        energy_joules = sliders['energy'].val
        weight = bb_weights[int(sliders['weight'].val)]
        mass_kg = weight / 1000.0
        spin_decay_rate = sliders['spin_decay'].val
        spin_rate = sliders['spin'].val
        target_distance = sliders['target'].val
        
        # Calculate initial velocity in m/s and fps
        initial_velocity_ms = np.sqrt((2 * energy_joules) / mass_kg)
        initial_velocity_fps = initial_velocity_ms * 3.28084  # Convert to FPS

        # Calculate extremes for combined tolerance
        tolerance = sliders['tolerance'].val
        weight_tol = sliders['weight_tolerance'].val / 100.0

        # Calculate all extreme combinations
        diameter_vars = [bb_diameter - tolerance, bb_diameter + tolerance]
        weight_vars = [weight * (1 - weight_tol), weight * (1 + weight_tol)]
        
        # Test all combinations of extremes to find true min/max
        extreme_trajectories = []
        for d in diameter_vars:
            for w in weight_vars:
                bb = BBTrajectory(
                    mass=w,
                    diameter=d,
                    initial_velocity=[initial_velocity_ms, 0, 0],
                    spin_rate=spin_rate,
                    spin_decay_rate=spin_decay_rate
                )
                times, positions = bb.simulate_trajectory()
                extreme_trajectories.append((times, positions))
        
        # Calculate nominal trajectory
        bb_nominal = BBTrajectory(
            mass=weight,
            diameter=bb_diameter,
            initial_velocity=[initial_velocity_ms, 0, 0],
            spin_rate=spin_rate,
            spin_decay_rate=spin_decay_rate
        )
        times_nominal, positions_nominal = bb_nominal.simulate_trajectory()
        
        # Find true min/max trajectories at each point and extend to ground
        def extend_to_ground(positions):
            # Find where trajectory hits ground (z=0)
            ground_idx = np.where(positions[:, 2] <= 0)[0]
            if len(ground_idx) > 0:
                x_ground = positions[ground_idx[0], 0]
                return x_ground
            return positions[-1, 0]

        # Get maximum x distance any trajectory reaches
        max_distance = max(extend_to_ground(traj[1]) for traj in extreme_trajectories)
        x_interp = np.linspace(0, max_distance, 1000)
        
        # Interpolate all trajectories and clamp to ground
        interpolated_heights = []
        for times, positions in extreme_trajectories:
            x_coords = positions[:, 0]
            z_coords = positions[:, 2]
            
            # Extend last point to ground if needed
            if z_coords[-1] > 0:
                x_ground = np.interp(0, z_coords[::-1], x_coords[::-1])
                x_coords = np.append(x_coords, x_ground)
                z_coords = np.append(z_coords, 0)
            
            interp_heights = np.interp(x_interp, x_coords, z_coords, right=0)
            interpolated_heights.append(interp_heights)
        
        # Find min and max at each point
        min_traj_interp = np.min(interpolated_heights, axis=0)
        max_traj_interp = np.max(interpolated_heights, axis=0)
        
        # Plot interpolated tolerance envelope
        plot_ax.fill_between(x_interp, min_traj_interp, max_traj_interp,
                           color='gray', alpha=0.2,
                           label=f'Tolerance (±{tolerance:.3f}mm, ±{sliders['weight_tolerance'].val:.1f}%)')

        # Use nominal trajectory for main calculations
        times, positions = times_nominal, positions_nominal

        # Calculate velocities for the nominal trajectory
        velocities = np.zeros(len(positions))
        for i in range(len(positions)-1):
            dp = positions[i+1] - positions[i]
            dt = times[i+1] - times[i]
            velocities[i] = np.linalg.norm(dp/dt)
        velocities[-1] = velocities[-2]
        
        # Store interpolated values for hover display
        format_coord.min_traj = min_traj_interp
        format_coord.max_traj = max_traj_interp
        format_coord.x_interp = x_interp
        
        # Store values needed for energy calculations
        format_coord.mass = mass_kg
        format_coord.positions = positions
        format_coord.velocities = velocities
        
        # ...rest of existing code until title...
        
        plot_ax.set_title(f'BB Trajectory Simulation\nEnergy: {energy_joules:.1f}J | {initial_velocity_fps:.0f} FPS', 
                         color='white', pad=20, fontsize=14)
        
        # ...rest of existing code...

        # Calculate velocity in FPS
        velocity_fps = velocities * 3.28084
        
        format_coord.positions = positions
        format_coord.velocities = velocities
        
        max_x = max(np.max(positions[:, 0]) * 1.1, target_distance * 1.1)
        
        # Draw optimal zone BEFORE the trajectory
        start_height = 1.5  # meters
        optimal_zone_height = 0.25  # 25cm in meters
        optimal_zone_top = start_height + optimal_zone_height
        optimal_zone_bottom = start_height - optimal_zone_height
        min_distance = 10  # Start optimal zone at 10 meters
        
        # Draw the zone (only after min_distance)
        plot_ax.axhspan(optimal_zone_bottom, optimal_zone_top,
                       xmin=min_distance/max_x,  # Start shading at 10m
                       color='green', alpha=0.2, zorder=1,
                       label='Optimal Zone (±25cm)')
        
        # Calculate where trajectory is within optimal zone (only after min_distance)
        mask_optimal = (positions[:, 2] <= optimal_zone_top) & \
                      (positions[:, 2] >= optimal_zone_bottom) & \
                      (positions[:, 0] >= min_distance)
                      
        if np.any(mask_optimal):
            optimal_start = max(positions[mask_optimal][0][0], min_distance)  # Start at min_distance
            optimal_end = positions[mask_optimal][-1][0]
            optimal_length = optimal_end - optimal_start
            
            # Add text showing optimal zone length
            plot_ax.text(optimal_start, optimal_zone_top + 0.1, 
                        f'Optimal zone: {optimal_length:.1f}m (from {optimal_start:.1f}m to {optimal_end:.1f}m)', 
                        color='white', fontsize=10, zorder=4)
            
            # Draw vertical lines at start and end of optimal zone
            plot_ax.axvline(x=optimal_start, color='green', linestyle='--', alpha=0.5, zorder=2)
            plot_ax.axvline(x=optimal_end, color='green', linestyle='--', alpha=0.5, zorder=2)
        
        # Ground line and target line with increased zorder
        plot_ax.axhline(y=0, color='white', linestyle='-', alpha=0.3, zorder=2)
        plot_ax.axvline(x=target_distance, color='red', linestyle='--', alpha=0.5, label='Target', zorder=2)
        
        target_idx = np.argmin(np.abs(positions[:, 0] - target_distance))
        time_to_target = times[target_idx]
        target_height = positions[target_idx, 2]
        
        # Update scatter plots with higher zorder to appear above the optimal zone
        plot_ax.scatter(target_distance, target_height, 
                       color='yellow', s=100, marker='*', zorder=3,
                       label=f'Time to target: {time_to_target:.3f}s\nHeight at target: {target_height:.2f}m')
        
        points = plot_ax.scatter(positions[:, 0], positions[:, 2], 
                               c=velocities, cmap='plasma',
                               s=1, label=f'{weight}g BB', zorder=3)
        
        plot_ax.scatter(positions[-1, 0], positions[-1, 2], 
                       color='red', s=50, marker='x', zorder=3,
                       label=f'Impact: {positions[-1, 0]:.1f}m ({positions[-1, 0]*3.28084:.1f}ft)')
        
        plot_ax.set_xlim(0, max_x)
        speed_ax.set_xlim(0, max_x)
        plot_ax.set_ylim(-0.1, np.max(positions[:, 2]) * 1.1)
        plot_ax.set_facecolor('#2F2F2F')
        
        cbar = plt.colorbar(points, cax=cbar_ax)
        cbar_ax.set_ylabel('Velocity (m/s)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        plot_ax.set_xlabel('Distance (m)', color='white')
        plot_ax.set_ylabel('Height (m)', color='white')
        plot_ax.set_title(f'BB Trajectory Simulation\nEnergy: {energy_joules:.1f}J | {initial_velocity_fps:.0f} FPS', 
                         color='white', pad=20, fontsize=14)
        plot_ax.grid(True, alpha=0.2, linestyle='--')
        plot_ax.legend(facecolor='#2F2F2F', edgecolor='#404040', framealpha=0.9,
                      title='Trajectory Info', title_fontsize=12)
        
        plot_ax.format_coord = format_coord
        
        plot_ax.tick_params(colors='white')
        
        cumulative_distance = np.zeros(len(positions))
        cumulative_distance[1:] = np.cumsum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))
        
        speeds = velocities
        
        speed_ax.plot(cumulative_distance, speeds, color='cyan')
        speed_ax.set_xlabel('Traveled Distance (m)', color='white')
        speed_ax.set_ylabel('Speed (m/s)', color='white')
        speed_ax.set_title('Speed over Traveled Distance', color='white')
        speed_ax.grid(True, alpha=0.2, linestyle='--')
        speed_ax.tick_params(colors='white')
        speed_ax.set_facecolor('#2F2F2F')
        
        textboxes['weight'].set_val(weight_format(sliders['weight'].val))
        
        fig.canvas.draw_idle()
    
    # Initialize status as ready
    status_text.set_text('Ready')
    status_text.set_color('lime')
    
    for slider in sliders.values():
        slider.on_changed(update)

    # Initial plot
    update(None)
    plt.show()

def main():
    bb_diameter = 5.95  # 6mm BB
    plot_interactive_trajectories(bb_diameter)

if __name__ == "__main__":
    main()
