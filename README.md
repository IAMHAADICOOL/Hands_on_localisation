# Hands-on Localisation — Part 2: Graph SLAM with A Priori Known Features

This branch implements **SLAM with a priori known features**: the map of landmark positions is fully provided upfront. The robot uses range+bearing observations of these known landmarks — in addition to compass heading and odometry — to jointly localise itself and refine landmark estimates via a GTSAM/ISAM2 factor graph.

---

## Concept

### Map-Based Localization (MBL) vs Full SLAM

When the map is known, the problem reduces to **localisation only**: the robot uses observations of map features to correct pose drift. There is no need to initialise new features or grow the state vector.

This part bridges the gap between pure localization (Part 1) and full SLAM (Part 3):
- Like Part 1, no new features are added during the run.
- Like Part 3, the full joint robot+landmark state is maintained and both pose and landmark uncertainties are tracked.

The state vector is fixed in dimension from the start:
```
xk = [ x, y, θ  |  f1x, f1y  |  f2x, f2y  |  ...  |  fNx, fNy ]
       robot pose          N known landmarks (initialised from map M)
```

### GTSAM Factor Graph

All observations create constraints in the factor graph. ISAM2 exploits the sparse structure of the graph to efficiently update only the affected parts of the trajectory on each new measurement. The joint optimisation over robot poses and landmark positions means observations to far-away landmarks propagate corrections back through the trajectory.

---

## Key Implementation: `FEKFSLAM.py`

### Constructor

```python
self.nf = len(self.M)  # all landmarks known at start
```

`nf` is fixed for the entire run. Compare to Part 1 (`nf = 0`) and Part 3 (`nf = 0`, then grows).

### `AddNewFeatures`

The method is **defined** (code identical to Part 3) but **never called** in the localization loop — it is commented out in `Localize`:

```python
# xk, Pk = self.AddNewFeatures(xk, Pk, znp, Rnp)
```

Non-paired observations (`znp`) are ignored since the map is assumed complete.

### `Prediction(uk, Qk, xk_1, Pk_1)`

Same block-matrix FEKFSLAM prediction as Part 3. The landmark blocks (`P_MM`, `P_RM`) are carried through unchanged while only the robot block is propagated by the motion model Jacobians.

Accumulated odometry between graph nodes:
```python
rel_pose = Pose3D(self.rel_disp)
self.rel_disp = rel_pose.oplus(Pose3D(uk))
self.rel_cov = J1_rel @ self.rel_cov @ J1_rel.T + J2_rel @ Qk @ J2_rel.T
```

### `Localize(xk_1, Pk_1, k)` — Sparse Pose Trigger

Same sparse trigger as Part 1: a new graph node is only created when a compass or feature measurement is available. The key difference from Part 1 is that `Update()` now receives `zf, Rf, self.H` (the feature observations and data association hypothesis):

```python
xk, Pk = self.Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk, k, xk_1, uk, Qk,
                     zf, Rf, self.H,           # ← feature obs (absent in Part 1)
                     pose_index=self.pose_index,
                     last_pose_step=self.last_pose_step,
                     accumulated_odom=self.accumulated_odom)
```

---

## Key Implementation: `EKF.py`

### `Update(...)` — ISAM2 Graph Update

In addition to the odometry and compass factors used in Part 1, this method now adds **bearing+range observation factors** for each matched landmark.

**Initial graph setup (once, at `pose_idx == 0`):**
```python
# Anchor pose
self.new_factors.add(gtsam.PriorFactorPose2(X(0), x0_pose, PRIOR_NOISE))

# Pre-load all known landmarks as priors
for j in range(len(self.M)):
    l_pos = gtsam.Point2(self.M[j][0], self.M[j][1])
    self.new_values.insert(L(j), l_pos)
    self.new_factors.add(gtsam.PriorFactorPoint2(L(j), l_pos, landmark_prior_noise))
```

**Each timestep:**

| Factor | GTSAM Type | Notes |
|--------|-----------|-------|
| Odometry | `BetweenFactorPose2(X(k), X(k+1))` | Noise from `rel_cov` |
| Compass | `PoseRotationPrior2D(X(k+1))` | Heading measurement |
| Landmark obs (per landmark) | `BearingRangeFactor2D(X(k+1), L(j))` | Bearing+range to associated landmark |

The bearing+range observation noise is computed by **propagating the Cartesian measurement covariance** `Rf` through the Cartesian-to-polar Jacobian:

```python
J = [[-y / r²,   x / r²],
     [ x / r,    y / r ]]
R_polar = J @ Rf_i @ J.T
br_noise = gtsam.noiseModel.Gaussian.Covariance(R_polar)
self.new_factors.add(gtsam.BearingRangeFactor2D(
    X(pose_idx+1), L(landmark_idx),
    gtsam.Rot2(bearing_val), range_val, br_noise))
```

**After ISAM2 update**, both pose and landmark positions are extracted:
```python
# Robot pose
pose_res = result.atPose2(X(pose_idx+1))

# Landmarks
for j in range(self.nf):
    l_res = result.atPoint2(L(j))
```

Marginal covariances are extracted for both the robot pose and each individual landmark via `gtsam.Marginals`, updating the full `Pk`.

---

## Data Association

`DataAssociation()` (from `FEKFMBL`) runs Mahalanobis-gated nearest-neighbour matching between current feature observations and expected observations of known map landmarks. Successfully paired observations go to `zf` (passed to `Update`); unmatched go to `znp` (discarded in Part 2).

---

## Running

```bash
python3 <main_simulation_script>.py
```

---

## Relationship to Other Parts

| | Part 1 | **Part 2** | Part 3 |
|-|--------|-----------|--------|
| Map at start | None | **All known** | None |
| `nf` at init | 0 | **`len(M)`** | 0 |
| `AddNewFeatures` called | Yes (no-op) | **No (commented out)** | Yes |
| Initial landmark GTSAM nodes | None | **`PriorFactorPoint2` for all** | Added dynamically |
| `BearingRangeFactor2D` | No | **Yes** | Yes |
| State vector size | Fixed (3) | **Fixed (3 + 2N)** | Grows over time |
| Landmark covariance updated | No | **Yes (via ISAM2 marginals)** | Yes |
