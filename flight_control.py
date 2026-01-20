import math
import random
from dataclasses import dataclass
from typing import List, Tuple
import jsbsim


EARTH_RADIUS_M = 6_378_137.0


@dataclass
class AircraftState:
	airspeed_mps: float
	geodetic_rad_m: Tuple[float, float, float]
	position_ned_m: Tuple[float, float, float]
	attitude_rad: Tuple[float, float, float]
	velocity_body_mps: Tuple[float, float, float]
	velocity_ned_mps: Tuple[float, float, float]
	angular_velocity_rps: Tuple[float, float, float]
	linear_accel_mps2: Tuple[float, float, float]
	angular_accel_rps2: Tuple[float, float, float]
	
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
		self.integral = _clip(self.integral + error * dt, -self.integ_limit, self.integ_limit)
		derivative = (error - self.prev_error) / dt if dt > 0.0 else 0.0
		self.prev_error = error
		output = self.kp * error + self.ki * self.integral + self.kd * derivative
		return _clip(output, -self.limit, self.limit)


def _wrap_pi(angle: float) -> float:
	while angle > math.pi:
		angle -= 2.0 * math.pi
	while angle < -math.pi:
		angle += 2.0 * math.pi
	return angle


def _clip(value: float, low: float, high: float) -> float:
	return max(low, min(high, value))

class UniController:
	"""Placeholder controller for future JSBSim integration."""

	def __init__(self) -> None:
		self.fdm = None
		self.aircraft = "x8"
		self.sim_hz = 30
		self.sim_dt = 1. / self.sim_hz
		self.reference_lat_rad = None
		self.reference_lon_rad = None
		self.reference_alt_m = None

	def initialize_jsbsim(self, root_dir: str) -> None:
		self.fdm = jsbsim.FGFDMExec(root_dir=root_dir)
		self.fdm.set_dt(self.sim_dt)
		if not self.fdm.load_model(self.aircraft):
			raise RuntimeError(f"Unable to load JSBSim aircraft '{self.aircraft}'.")

	def initialize_state(self) -> None:
		if self.fdm is None:
			raise RuntimeError("JSBSim not initialized.")
		mps_to_kts = 1.9438444924406048
		meter_to_ft = 1.0 / 0.3048
		self.fdm['ic/latitude-deg'] = 0.0
		self.fdm['ic/longitude-deg'] = 0.0
		self.fdm['ic/h-sl-ft'] = 10.0 * meter_to_ft
		self.fdm['ic/phi-deg'] = 0.0
		self.fdm['ic/theta-deg'] = 0.0
		self.fdm['ic/psi-true-deg'] = 0.0
		self.fdm['ic/u-fps'] = 16.0 * meter_to_ft
		self.fdm['ic/v-fps'] = 0.0
		self.fdm['ic/w-fps'] = 0.0
		self.fdm['ic/p-rad_sec'] = 0.0
		self.fdm['ic/q-rad_sec'] = 0.0
		self.fdm['ic/r-rad_sec'] = 0.0
		self.fdm['ic/vc-kts'] = 16.0 * mps_to_kts
		if not self.fdm.run_ic():
			raise RuntimeError("JSBSim failed to apply initial conditions.")
		self.reference_lat_rad = self.fdm['position/lat-gc-rad']
		self.reference_lon_rad = self.fdm['position/long-gc-rad']
		self.reference_alt_m = self.fdm['position/h-sl-ft'] * 0.3048

	def get_aircraft_state(self) -> AircraftState:
		ft_to_meter = 0.3048
		# airspeed [m/s] 
		airspeed = self.fdm["velocities/vt-fps"] * ft_to_meter
		# latitude, longitude, altitude [rad, rad, m] 
		lat = self.fdm["position/lat-gc-rad"]
		lon = self.fdm["position/long-gc-rad"]
		altitude = self.fdm["position/h-sl-ft"] * ft_to_meter
		# NED position [m, m, m] 
		if self.reference_lat_rad is None:
			self.reference_lat_rad = lat
			self.reference_lon_rad = lon
			self.reference_alt_m = altitude
		cos_ref = math.cos(self.reference_lat_rad)
		if abs(cos_ref) < 1e-6:
			cos_ref = math.copysign(1e-6, cos_ref if cos_ref != 0.0 else 1.0)
		north = (lat - self.reference_lat_rad) * EARTH_RADIUS_M
		east = (lon - self.reference_lon_rad) * EARTH_RADIUS_M * cos_ref
		down = -altitude
		# attitude [rad, rad, rad] 
		roll = self.fdm["attitude/phi-rad"]
		pitch = self.fdm["attitude/theta-rad"]
		yaw = self.fdm["attitude/psi-rad"]
		# velocity body [m/s, m/s, m/s] 
		u = self.fdm["velocities/u-fps"] * ft_to_meter
		v = self.fdm["velocities/v-fps"] * ft_to_meter
		w = self.fdm["velocities/w-fps"] * ft_to_meter
		# velocity NED [m/s, m/s, m/s] 
		vel_north = self.fdm["velocities/v-north-fps"] * ft_to_meter
		vel_east = self.fdm["velocities/v-east-fps"] * ft_to_meter
		vel_down = self.fdm["velocities/v-down-fps"] * ft_to_meter
		# velocity angular [rad/s, rad/s, rad/s] 
		p = self.fdm["velocities/p-rad_sec"]
		q = self.fdm["velocities/q-rad_sec"]
		r = self.fdm["velocities/r-rad_sec"]
		# accelerations body [m/s^2, m/s^2, m/s^2] 
		udot = self.fdm["accelerations/udot-ft_sec2"] * ft_to_meter
		vdot = self.fdm["accelerations/vdot-ft_sec2"] * ft_to_meter
		wdot = self.fdm["accelerations/wdot-ft_sec2"] * ft_to_meter
		# accelerations angular [rad/s^2, rad/s^2, rad/s^2] 
		pdot = self.fdm["accelerations/pdot-rad_sec2"]
		qdot = self.fdm["accelerations/qdot-rad_sec2"]
		rdot = self.fdm["accelerations/rdot-rad_sec2"]
		# ----------------------------------------------------------

		return AircraftState(
			airspeed_mps=airspeed,
			geodetic_rad_m=(lat, lon, altitude),
			position_ned_m=(north, east, down),
			attitude_rad=(roll, pitch, yaw),
			velocity_body_mps=(u, v, w),
			velocity_ned_mps=(vel_north, vel_east, vel_down),
			angular_velocity_rps=(p, q, r),
			linear_accel_mps2=(udot, vdot, wdot),
			angular_accel_rps2=(pdot, qdot, rdot)
			)
	
	def pitch_hold(self, target_pitch_rad: float, current_pitch_rad: float, dt: float) -> float:
		error = _wrap_pi(target_pitch_rad - current_pitch_rad)
		kp = 2.0
		ki = 0.5
		kd = 0.1
		limit = 1.0
		integ_limit = 0.2
		if not hasattr(self, 'pitch_pid'):
			self.pitch_pid = PIDController(kp, ki, kd, limit, integ_limit)
		# JSBSim elevator command positive drives nose down, hence the sign flip.
		return _clip(-self.pitch_pid.update(error, dt), -limit, limit)
	
	def roll_hold(self, target_roll_rad: float, current_roll_rad: float, dt: float) -> float:
		error = _wrap_pi(target_roll_rad - current_roll_rad)
		kp = 2.0
		ki = 0.5
		kd = 0.1
		limit = 1.0
		integ_limit = 0.2
		if not hasattr(self, 'roll_pid'):
			self.roll_pid = PIDController(kp, ki, kd, limit, integ_limit)
		return _clip(self.roll_pid.update(error, dt), -limit, limit)
	
	def altitude_hold(self, target_altitude_m: float, current_altitude_m: float, dt: float) -> float:
		error = target_altitude_m - current_altitude_m
		kp = 0.04
		ki = 0.01
		kd = 0.0
		pitch_limit = math.radians(20.0)
		integ_limit = 50.0
		if not hasattr(self, 'altitude_pid'):
			self.altitude_pid = PIDController(kp, ki, kd, pitch_limit, integ_limit)
		target_pitch_rad = self.altitude_pid.update(error, dt)
		current_pitch_rad = self.fdm["attitude/theta-rad"]
		return self.pitch_hold(target_pitch_rad, current_pitch_rad, dt)
	
	def heading_hold(self, target_yaw_rad: float, current_yaw_rad: float, dt: float) -> float:
		error = _wrap_pi(target_yaw_rad - current_yaw_rad)
		kp = 1.5
		ki = 0.2
		kd = 0.05
		roll_limit = math.radians(25.0)
		integ_limit = math.radians(60.0)
		if not hasattr(self, 'heading_pid'):
			self.heading_pid = PIDController(kp, ki, kd, roll_limit, integ_limit)
		target_roll_rad = self.heading_pid.update(error, dt)
		current_roll_rad = self.fdm["attitude/phi-rad"]
		return self.roll_hold(target_roll_rad, current_roll_rad, dt)
	
	def yaw_rate_hold(self, target_yaw_rate_rps: float, current_yaw_rate_rps: float, dt: float) -> float:
		error = target_yaw_rate_rps - current_yaw_rate_rps
		kp = 1.0
		ki = 0.1
		kd = 0.02
		roll_limit = math.radians(30.0)
		integ_limit = math.radians(45.0)
		if not hasattr(self, 'yaw_rate_pid'):
			self.yaw_rate_pid = PIDController(kp, ki, kd, roll_limit, integ_limit)
		target_roll_rad = self.yaw_rate_pid.update(error, dt)
		current_roll_rad = self.fdm["attitude/phi-rad"]
		return self.roll_hold(target_roll_rad, current_roll_rad, dt)
	
	def climb_rate_hold(self, target_climb_rate_mps: float, current_climb_rate_mps: float, dt: float) -> float:
		error = target_climb_rate_mps - current_climb_rate_mps
		kp = 0.1
		ki = 0.03
		kd = 0.02
		pitch_limit = math.radians(25.0)
		integ_limit = 40.0
		if not hasattr(self, 'climb_rate_pid'):
			self.climb_rate_pid = PIDController(kp, ki, kd, pitch_limit, integ_limit)
		target_pitch_rad = self.climb_rate_pid.update(error, dt)
		current_pitch_rad = self.fdm["attitude/theta-rad"]
		return self.pitch_hold(target_pitch_rad, current_pitch_rad, dt)
	
	def airspeed_hold(self, target_airspeed_mps: float, current_airspeed_mps: float, dt: float) -> float:
		error = target_airspeed_mps - current_airspeed_mps
		kp = 0.15
		ki = 0.02
		kd = 0.01
		limit = 1.0
		integ_limit = 0.4
		if not hasattr(self, 'airspeed_pid'):
			self.airspeed_pid = PIDController(kp, ki, kd, limit, integ_limit)
		throttle_cmd = self.airspeed_pid.update(error, dt)
		return _clip(throttle_cmd, 0.0, 1.0)

	def control_actuators(self, controls: Tuple[float, float, float]) -> None:
		throttle, aileron, elevator = controls
		self.fdm["fcs/throttle-cmd-norm"] = throttle
		self.fdm["fcs/aileron-cmd-norm"] = aileron
		self.fdm["fcs/elevator-cmd-norm"] = elevator
	
	def step(self) -> None:
		self.fdm.run()

def main() -> None:
	controller = UniController()
	controller.initialize_jsbsim(root_dir=None)
	controller.initialize_state()
	sim_time = controller.fdm.get_sim_time()
	sim_time_total = 20.0
	while sim_time < sim_time_total:
		sim_time = controller.fdm.get_sim_time()
		state = controller.get_aircraft_state()
		
		throttle_cmd = controller.airspeed_hold(20.0, state.airspeed_mps, controller.sim_dt)
		aileron_cmd = controller.heading_hold(0., state.attitude_rad[2], controller.sim_dt)
		elevator_cmd = controller.climb_rate_hold(1.0, -state.velocity_ned_mps[2], controller.sim_dt)
		controller.control_actuators((throttle_cmd, aileron_cmd, elevator_cmd))

		print(f"T: {sim_time:.2f} s, "
            f"AS: {state.airspeed_mps:.2f} m/s, "
			f"P(NED): [{state.position_ned_m[0]:5.1f}, {state.position_ned_m[1]:5.1f}, {state.position_ned_m[2]:5.1f}] m, ")
		if state.angular_velocity_rps[0] > 5 * math.pi:
			print("Unrealistic angular velocity detected, terminating simulation.")
			break
		controller.step()

	
if __name__ == "__main__":
	main()