# Hands-on Localisation — Part 1: Graph SLAM for Localization with Compass

This branch implements **EKF-based robot localization** augmented with a **GTSAM/ISAM2 factor graph** for global trajectory smoothing. The robot has no landmarks in its state — only its own pose `[x, y, θ]` — and uses a compass sensor to correct heading drift from dead-reckoning odometry.

---

## Concept

### The Localization Problem

The robot moves in a 2D plane using a differential-drive model. Wheel odometry integrates velocities to estimate pose, but accumulates drift over time. A compass provides periodic absolute heading measurements that can correct this drift.

The standard EKF approach:
1. **Predict** the next pose using the motion model (odometry)
2. **Update** the pose estimate using the compass measurement

### Why a Factor Graph?

A standard EKF update uses only the *current* measurement to correct the *current* state. A factor graph (GTSAM/ISAM2) treats the entire trajectory as a joint optimisation problem. Each new measurement adds a constraint to the graph, and ISAM2 efficiently re-solves the full trajectory, correcting **past** poses as well as the current one. This provides:

- **Loop closure ready**: any future constraint (e.g. revisiting a known location) can correct the whole past trajectory
- **Global consistency**: accumulated drift is distributed across the trajectory rather than instantaneously snapped at each correction

### Sparse Pose Graph

Not every timestep creates a pose node in the graph. A new graph node `X(k)` is only added when a compass reading is available. Between compass readings, odometry is **accumulated** via `rel_disp` and `rel_cov` (using compound pose composition `Pose3D.oplus`) so that the `BetweenFactorPose2` edge captures the total integrated displacement between nodes rather than per-step increments.

---

## Key Implementation: `EKF.py`

### `Update(...)` — ISAM2 Graph Update

This method replaces the standard EKF update step. The Kalman gain / innovation correction is commented out; all state estimation is done by ISAM2.

**Graph structure at each new pose node:**

| Factor | GTSAM Type | Noise Source |
|--------|-----------|-------------|
| Initial anchor (once) | `PriorFactorPose2(X(0))` | Fixed: `diag([0.3, 0.3, 0.1])` |
| Odometry edge | `BetweenFactorPose2(X(k), X(k+1))` | `rel_cov` — propagated accumulated odometry covariance |
| Compass | `PoseRotationPrior2D(X(k+1), Rot2(θ))` | `sqrt(Rk[0,0])` from sensor model |

```python
# Anchor at first pose
self.new_factors.add(gtsam.PriorFactorPose2(X(0), x0_pose, PRIOR_NOISE))

# Odometry between consecutive graph nodes
self.new_factors.add(gtsam.BetweenFactorPose2(
    X(pose_idx), X(pose_idx+1), odom_to_use, odometry_noise))

# Compass heading
self.new_factors.add(gtsam.PoseRotationPrior2D(
    X(pose_idx+1), gtsam.Rot2(compass_yaw), compass_noise))
```

**After ISAM2 update**, the optimised pose is extracted and written back:
```python
pose_res = result.atPose2(X(pose_idx + 1))
self.xk = np.array([[pose_res.x()], [pose_res.y()], [pose_res.theta()]])
```
The pose marginal covariance is also extracted via `gtsam.Marginals` and replaces `Pk`.

**No landmark nodes** are added to the graph in this part. The landmark measurement and GTSAM `PriorFactorPoint2` blocks are fully commented out.

---

## Key Implementation: `FEKFSLAM.py`

### `Prediction(uk, Qk, xk_1, Pk_1)`

Full FEKFSLAM block-matrix prediction, implemented to support the joint robot+map state structure even though in Part 1 the map is empty:

```
P_bar = [ Jfx @ P_RR @ Jfx.T + Jfw @ Qk @ Jfw.T  |  Jfx @ P_RM ]
         [ (Jfx @ P_RM).T                           |  P_MM       ]
```

The accumulated odometry trackers are updated here:
```python
rel_pose = Pose3D(self.rel_disp)
self.rel_disp = rel_pose.oplus(Pose3D(uk))
self.rel_cov = J1_rel @ self.rel_cov @ J1_rel.T + J2_rel @ Qk @ J2_rel.T
```

### `Localize(xk_1, Pk_1, k)` — Sparse Pose Trigger

A new graph node is only added when a compass reading exists (`zm is not None`):

```python
if zk is not None:  # compass reading available
    xk, Pk = self.Update(zk, Rk, xk_bar, Pk_bar, ..., pose_index=self.pose_index)
    self.pose_index += 1
    self.rel_disp = np.zeros(...)   # reset accumulator
    self.rel_cov = np.zeros(...)
else:
    # No compass: skip ISAM2, just propagate EKF prediction
    xk, Pk = xk_bar, Pk_bar
```

### `AddNewFeatures`

Called at the end of each iteration (features are observed from the beginning), but since `nf = 0` and no features are provided, this is effectively a no-op in Part 1.

---

## State Summary

| Variable | Content |
|----------|---------|
| `xk` | `[x, y, θ]` — robot pose only |
| `Pk` | 3×3 pose covariance |
| `pose_index` | Count of ISAM2 graph nodes so far |
| `rel_disp` | Accumulated odometry since last graph node |
| `rel_cov` | Propagated covariance of `rel_disp` |

---

## Running

```bash
python3 <main_simulation_script>.py
```

---

## Relationship to Other Parts

| | Part 1 | Part 2 | Part 3 |
|-|--------|--------|--------|
| Sensor used | Compass (heading) | Range/bearing to landmarks | Range/bearing + compass |
| Landmarks in state | No | Yes (pre-given) | Yes (discovered on-line) |
| GTSAM landmark nodes | No | `PriorFactorPoint2` for all | `PriorFactorPoint2` added dynamically |
| `BearingRangeFactor2D` | No | Yes | Yes |
| Map expansion | No | No | Yes — state vector grows |
