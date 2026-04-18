# UAV Cheat Sheet

## Aircraft Axes

Imagine sitting in the cockpit looking forward:

- **X axis:** Runs nose to tail (front to back)
- **Y axis:** Runs wingtip to wingtip (left to right)
- **Z axis:** Runs top to bottom (up and down)

## Roll, Pitch, Yaw

- **Roll:** The drone tilts its wings -- one goes up, the other goes down. Rotates around the X axis. This is how it turns.
- **Pitch:** The nose goes up or down. Rotates around the Y axis. This is how it climbs or descends.
- **Yaw:** The nose swings left or right, like a compass needle. Rotates around the Z axis. This is the drone changing its heading direction.

## Quaternions

A way to describe which direction the drone is pointing using four numbers instead of roll/pitch/yaw angles. Flight controllers use them because roll/pitch/yaw has a math problem called "gimbal lock" where certain angle combinations break calculations. Quaternions avoid this.

All four values range between -1 and 1.

| Component | What it represents | In our data |
|-----------|-------------------|-------------|
| q[0] (w) | How much of the rotation is "none" -- closer to 1 means barely rotated, closer to 0 means a big rotation | Wide range -- heading changes a lot on the circuit |
| q[1] (x) | How much rotation is around the nose-to-tail axis (related to roll) | Tight range -- drone doesn't roll aggressively |
| q[2] (y) | How much rotation is around the wingtip axis (related to pitch) | Tight range -- drone doesn't pitch dramatically |
| q[3] (z) | How much rotation is around the up-down axis (related to yaw/heading) | Wide range -- drone constantly changes heading on the circuit |

## PWM (Pulse Width Modulation)

How the flight controller talks to the motor and control surfaces. The number (1000-2000) is the width of an electrical pulse in microseconds. Higher number = more throttle or more deflection.

| Value | Meaning |
|-------|---------|
| 1000 | Minimum (off or neutral) |
| 1500 | Midpoint |
| 2000 | Maximum (full throttle or full deflection) |

## Control Surfaces

| Surface | What it does | Controlled by |
|---------|-------------|---------------|
| Throttle | Controls motor speed / power | output[0]_f3 |
| Ailerons | Tilt the wings for turning (roll) | output[1]_f3 |
| Elevator | Tilt the nose up/down (pitch) | output[2]_f3 |
| Rudder | Swing the nose left/right (yaw) | output[3]_f3 |

## IMU (Inertial Measurement Unit)

Two sensors in one package:

- **Gyroscope:** Measures rotation rates (how fast the drone is spinning around each axis). Units: radians per second.
- **Accelerometer:** Measures forces acting on the drone along each axis. Units: meters per second squared. The z-axis reads about -9.8 m/s2 in level flight because of gravity.

## Energy Efficiency Metric

From the paper:

- **Power:** voltage (V) x current (A) = watts
- **Ground speed:** sqrt(vx^2 + vy^2 + vz^2) = meters per second
- **Efficiency:** ground speed / power = meters per watt
- **Threshold:** Below 0.1 m/W = inefficient
