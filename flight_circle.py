"""Circle flight controller for the Skywalker X8 using JSBSim.

The script builds a lightweight PID-based autopilot that keeps the
aircraft flying a level circle with constant airspeed. Tune the control
gains in CIRCLE_CONFIG to suit your JSBSim aircraft definition.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import jsbsim


G_ACCEL = 9.80665
EARTH_RADIUS_M = 6_378_137.0


def wrap_angle(angle: float) -> float:
	"""Wrap angle to [-pi, pi]."""

	return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, lower: float, upper: float) -> float:
	"""Clamp value into [lower, upper]."""

	return max(lower, min(value, upper))


def safe_cos(angle: float, eps: float = 1e-6) -> float:
	"""Return cos(angle) with a floor on magnitude to avoid divide-by-zero."""

	value = math.cos(angle)
	if abs(value) < eps:
		value = math.copysign(eps, value if value != 0.0 else 1.0)
	return value


class PIDController:
	"""Simple PID controller with clamping."""

	def __init__(self, kp: float, ki: float, kd: float, limit: float = 1.0, integ_limit: float = 0.5):
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self.limit = limit
		self.integ_limit = integ_limit
		self.integral = 0.0
		self.prev_error = 0.0

	def reset(self) -> None:
		self.integral = 0.0
		self.prev_error = 0.0

	def update(self, error: float, dt: float) -> float:
		self.integral = clamp(self.integral + error * dt, -self.integ_limit, self.integ_limit)
		derivative = (error - self.prev_error) / dt if dt > 0.0 else 0.0
		self.prev_error = error
		output = self.kp * error + self.ki * self.integral + self.kd * derivative
		return clamp(output, -self.limit, self.limit)


@dataclass
class CircleConfig:
	aircraft: str = "SkywalkerX8"
	radius_m: float = 500.0
	target_alt_m: float = 100.0
	target_v_ms: float = 25.0
	circle_direction: int = -1  # 1 for counter clockwise, -1 for clockwise
	time_step: float = 0.01
	duration_s: float = 300.0
	bank_limit_deg: float = 45.0
	pitch_limit_deg: float = 20.0
	plot_enabled: bool = False
	orbit_gain: float = 2.0
	straight_heading_gain: float = 2.0
	join_threshold_m: float = 50.0
	reference_lat_deg: float = 0.0
	reference_lon_deg: float = 0.0
	initial_position_ned: Tuple[float, float, float] = (0.0, 0.0, -30.0)
	initial_attitude_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
	circle_center_ned: Tuple[float, float, float] = (0.0, 0.0, -100.0)


class CircleFlightController:
	def __init__(self, root_dir: str, config: CircleConfig):
		self.config = config
		self.reference_lat_rad = math.radians(config.reference_lat_deg)
		self.reference_lon_rad = math.radians(config.reference_lon_deg)
		self.fdm = jsbsim.FGFDMExec(root_dir=root_dir)
		self.fdm.set_dt(config.time_step)
		self._load_model(config.aircraft)
		self._setup_initial_conditions()
		self._init_autopilot()
		self._compute_circle_center()
		self.track_east = []
		self.track_north = []
		self.initial_heading = self.fdm['attitude/psi-rad']
		self.in_orbit = False
		self.join_time = None

	def _get_lat_lon_alt(self) -> Tuple[float, float, float]:
		lat = self.fdm["position/lat-gc-rad"]
		lon = self.fdm["position/long-gc-rad"]
		alt = self.fdm["position/h-sl-ft"] * 0.3048
		return lat, lon, alt

	def _load_model(self, aircraft: str) -> None:
		if not self.fdm.load_model(aircraft):
			raise RuntimeError(f"Unable to load JSBSim aircraft '{aircraft}'.")

	def _setup_initial_conditions(self) -> None:
		cfg = self.config
		north, east, down = cfg.initial_position_ned
		lat, lon = self._ned_to_geodetic(north, east)
		altitude_m = -down
		yaw_deg, pitch_deg, roll_deg = cfg.initial_attitude_deg
		self.fdm['ic/h-sl-m'] = altitude_m
		self.fdm['ic/vc-kts'] = cfg.target_v_ms * 1.94384
		self.fdm['ic/psi-true-rad'] = math.radians(yaw_deg)
		self.fdm['ic/theta-rad'] = math.radians(pitch_deg)
		self.fdm['ic/phi-rad'] = math.radians(roll_deg)
		self.fdm['ic/latitude-deg'] = math.degrees(lat)
		self.fdm['ic/longitude-deg'] = math.degrees(lon)
		self.fdm['ic/heading-true-deg'] = yaw_deg
		if not self.fdm.run_ic():
			raise RuntimeError("JSBSim failed to run initial conditions.")

	def _init_autopilot(self) -> None:
		cfg = self.config
		self.max_bank_rad = math.radians(cfg.bank_limit_deg)
		self.max_pitch_rad = math.radians(cfg.pitch_limit_deg)
		self.roll_pid = PIDController(kp=1.0, ki=0.0, kd=0.8, limit=1.0)
		self.altitude_pid = PIDController(kp=0.02, ki=0.002, kd=0.04, limit=self.max_pitch_rad)
		self.pitch_pid = PIDController(kp=2.5, ki=0.1, kd=0.3, limit=0.6)
		self.speed_pid = PIDController(kp=0.1, ki=0.05, kd=0.02, limit=1.0)

	def _compute_circle_center(self) -> None:
		north_c, east_c, down_c = self.config.circle_center_ned
		self.center_lat = self.reference_lat_rad + north_c / EARTH_RADIUS_M
		self.center_lon = self.reference_lon_rad + east_c / (EARTH_RADIUS_M * safe_cos(self.reference_lat_rad))
		self.center_alt_m = -down_c
		self.center_lat_cos = safe_cos(self.center_lat)

	def _ned_to_geodetic(self, north: float, east: float) -> Tuple[float, float]:
		lat = self.reference_lat_rad + north / EARTH_RADIUS_M
		lon = self.reference_lon_rad + east / (EARTH_RADIUS_M * safe_cos(self.reference_lat_rad))
		return lat, lon

	def _render_plot(self) -> None:
		if not self.config.plot_enabled:
			return
		try:
			import matplotlib.pyplot as plt
		except ModuleNotFoundError as exc:
			raise RuntimeError("Matplotlib is required for --plot; install it or run without --plot.") from exc
		if not self.track_east or not self.track_north:
			return
		radius = self.config.radius_m
		pad = max(100.0, radius * 0.2)
		limit = radius + pad
		fig, ax = plt.subplots()
		ax.set_title("Skywalker X8 circle track")
		ax.set_xlabel("East offset [m]")
		ax.set_ylabel("North offset [m]")
		ax.set_aspect("equal", adjustable="box")
		ax.grid(True, linestyle="--", linewidth=0.5)
		ax.set_xlim(-limit, limit)
		ax.set_ylim(-limit, limit)
		angles = [i * math.pi / 90.0 for i in range(181)]
		circle_east = [radius * math.sin(angle) for angle in angles]
		circle_north = [radius * math.cos(angle) for angle in angles]
		ax.plot(circle_east, circle_north, "k--", linewidth=1.0, label="Target circle")
		ax.plot(self.track_east, self.track_north, "b-", linewidth=1.5, label="Flight path")
		ax.plot(self.track_east[-1], self.track_north[-1], "ro", markersize=4, label="End point")
		ax.legend(loc="upper right")
		plt.show()

	def _get_state(self) -> Tuple[float, float, float, float, float, float, float]:
		lat, lon, altitude = self._get_lat_lon_alt()
		rel_north = (lat - self.center_lat) * EARTH_RADIUS_M
		rel_east = (lon - self.center_lon) * EARTH_RADIUS_M * self.center_lat_cos
		heading = self.fdm["attitude/psi-rad"]
		roll = self.fdm["attitude/phi-rad"]
		pitch = self.fdm["attitude/theta-rad"]
		speed = self.fdm['velocities/vt-fps'] * 0.3048
		return rel_north, rel_east, altitude, heading, roll, pitch, speed

	def _apply_controls(self, aileron: float, elevator: float, throttle: float) -> None:
		self.fdm['fcs/aileron-cmd-norm'] = aileron
		self.fdm['fcs/elevator-cmd-norm'] = elevator
		self.fdm['fcs/rudder-cmd-norm'] = 0.0
		self.fdm['fcs/throttle-cmd-norm[0]'] = throttle

	def _guidance(self, dt: float) -> None:
		cfg = self.config
		rel_north, rel_east, altitude, heading, roll, pitch, speed = self._get_state()

		radius = math.hypot(rel_north, rel_east)
		radial_error = cfg.radius_m - radius
		radial_heading = math.atan2(rel_east, rel_north)
		orbit_heading = radial_heading + cfg.circle_direction * math.pi / 2.0
		if not self.in_orbit and abs(radial_error) <= cfg.join_threshold_m:
			self.in_orbit = True
			self.join_time = self.fdm.get_sim_time()

		if self.in_orbit:
			heading_error = wrap_angle(orbit_heading - heading)
			base_bank = math.atan2(speed * speed, G_ACCEL * cfg.radius_m) * cfg.circle_direction
			target_bank = base_bank + cfg.orbit_gain * heading_error
		else:
			heading_error = wrap_angle(self.initial_heading - heading)
			target_bank = cfg.straight_heading_gain * heading_error

		target_bank = clamp(target_bank, -self.max_bank_rad, self.max_bank_rad)
		bank_error = target_bank - roll
		aileron = self.roll_pid.update(bank_error, dt)

		target_pitch = self.altitude_pid.update(cfg.target_alt_m - altitude, dt)
		target_pitch = clamp(target_pitch, -self.max_pitch_rad, self.max_pitch_rad)
		pitch_error = target_pitch - pitch
		elevator = -self.pitch_pid.update(pitch_error, dt)

		throttle = self.speed_pid.update(cfg.target_v_ms - speed, dt)
		throttle = clamp(throttle, 0.0, 1.0)

		self._apply_controls(aileron, elevator, throttle)

	def run(self) -> None:
		duration = self.config.duration_s
		dt = self.config.time_step
		report_interval = 1.0
		next_report = 0.0
		sim_time = 0.0
		while sim_time < duration and self.fdm.run():
			self._guidance(dt)
			sim_time = self.fdm.get_sim_time()
			rel_north, rel_east, altitude, heading, roll, pitch, speed = self._get_state()
			self.track_east.append(rel_east)
			self.track_north.append(rel_north)
			if sim_time >= next_report:
				radius_error = abs(math.hypot(rel_north, rel_east) - self.config.radius_m)
				print(
					f"t={sim_time:5.1f}s alt={altitude:4.1f}m v={speed:4.1f}m/s "
					f"hdg={math.degrees(heading):4.1f}deg roll={math.degrees(roll):2.1f}deg "
					f"radius_err={radius_error:3.1f}m"
				)
				next_report += report_interval
		self._render_plot()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Circle flight controller for Skywalker X8 in JSBSim")
	parser.add_argument("--root", default="/home/running-mars/.conda/envs/sim/lib/python3.8/site-packages/jsbsim", help="Path to the JSBSim root directory (contains aircraft/, engine/, systems/)")
	parser.add_argument("--aircraft", default="x8", help="JSBSim aircraft name to load")
	parser.add_argument("--radius", type=float, default=500.0, help="Desired circle radius in meters")
	parser.add_argument("--altitude", type=float, default=100.0, help="Target altitude in meters")
	parser.add_argument("--speed", type=float, default=20.0, help="Target true airspeed in m/s")
	parser.add_argument("--direction", type=int, choices=[-1, 1], default=-1, help="Circle direction: 1 CCW, -1 CW")
	parser.add_argument("--orbit-gain", type=float, default=2.0, help="Gain for radial error shaping during orbit tracking")
	parser.add_argument("--straight-heading-gain", type=float, default=2.0, help="Gain that converts heading error to bank during the approach phase")
	parser.add_argument("--join-threshold", type=float, default=50.0, help="Enter orbit when |radius error| falls below this threshold (meters)")
	parser.add_argument("--reference-lat", type=float, default=0.0, help="Reference latitude for the local NED frame (degrees)")
	parser.add_argument("--reference-lon", type=float, default=0.0, help="Reference longitude for the local NED frame (degrees)")
	parser.add_argument("--initial-north", type=float, default=0.0, help="Initial north offset in the NED frame (meters)")
	parser.add_argument("--initial-east", type=float, default=0.0, help="Initial east offset in the NED frame (meters)")
	parser.add_argument("--initial-down", type=float, default=-30.0, help="Initial down offset in the NED frame (meters)")
	parser.add_argument("--center-north", type=float, default=0.0, help="Circle center north offset in the NED frame (meters)")
	parser.add_argument("--center-east", type=float, default=0.0, help="Circle center east offset in the NED frame (meters)")
	parser.add_argument("--center-down", type=float, default=-100.0, help="Circle center down offset in the NED frame (meters)")
	parser.add_argument("--duration", type=float, default=200.0, help="Simulation duration in seconds")
	parser.add_argument("--time-step", type=float, default=0.01, help="Integrator time step in seconds")
	parser.add_argument("--plot", action="store_true", help="Render Matplotlib track plot once the simulation finishes")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = CircleConfig(
		aircraft=args.aircraft,
		radius_m=args.radius,
		target_alt_m=args.altitude,
		target_v_ms=args.speed,
		circle_direction=args.direction,
		duration_s=args.duration,
		time_step=args.time_step,
		plot_enabled=args.plot,
		orbit_gain=args.orbit_gain,
		straight_heading_gain=args.straight_heading_gain,
		join_threshold_m=args.join_threshold,
		reference_lat_deg=args.reference_lat,
		reference_lon_deg=args.reference_lon,
		initial_position_ned=(args.initial_north, args.initial_east, args.initial_down),
		circle_center_ned=(args.center_north, args.center_east, args.center_down),
	)
	controller = CircleFlightController(root_dir=args.root, config=config)
	controller.run()


if __name__ == "__main__":
	main()

