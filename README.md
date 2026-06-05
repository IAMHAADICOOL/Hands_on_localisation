# Hands-on Localisation — Part 3: Full Graph SLAM (Unknown Features)

This branch implements **Full SLAM with unknown features**: the robot simultaneously builds a map of its environment from scratch and localises itself within it, with no prior knowledge of landmark positions.

---

## Concept

In Full SLAM, the map is not given — the robot must discover and track landmarks on-the-fly. Each time a new landmark is observed for the first time (no matching feature in the current map), it is **initialised** into the state vector and the factor graph. Subsequent observations of the same landmark then constrain both the robot pose and the landmark position jointly, progressively reducing uncertainty in both.

The state vector starts as a pure robot pose and grows dynamically:

```
xk = [ x, y, θ  |  f1x, f1y  |  f2x, f2y  | ... ]
       robot pose   landmark 1    landmark 2
```

The covariance matrix grows correspondingly with each new feature added.

### EKF + GTSAM/ISAM2 Architecture

This lab uses a **hybrid** approach:

- The **EKF** handles the local prediction step and state augmentation mechanics (Jacobians, covariance growth on new features).
- **GTSAM ISAM2** performs global smoothing over the full robot trajectory and landmark positions via an incremental factor graph.

Each time a measurement is available, the graph is updated and ISAM2 re-optimises the entire trajectory. The EKF state and covariance are then overwritten with the ISAM2 result.

---

## Key Implementation: `FEKFSLAM.py`

### `AddNewFeatures(xk, Pk, znp, Rnp)`

Called at every timestep with the vector of **non-paired observations** — feature sightings that did not match any existing map entry.

For each new feature `zi`:

1. **Initialise position** in world frame using the inverse observation function:
   ```python
   x_new_feature = self.g(x_B, self.Feature(zi))
   ```

2. **Grow the covariance matrix** using Jacobians of `g()` w.r.t. robot pose (`J1`) and observation noise (`J2`):
   ```
   P_new_new = J1 @ P_B @ J1.T + J2 @ Ri @ J2.T
   P_new_old = J1 @ P_robot_all        # cross-correlations with everything
   ```
   The new block-augmented covariance is assembled as:
   ```
   Pk_plus = [ Pk_old      P_new_old.T ]
              [ P_new_old   P_new_new   ]
   ```

3. **Register in GTSAM**: the new landmark is immediately inserted as a `PriorFactorPoint2` node with covariance `P_new_new`, and ISAM2 is updated so the landmark is available for future measurement constraints.

`self.nf` starts at **0** and increments with every new feature added.

### `Prediction(uk, Qk, xk_1, Pk_1)`

Standard FEKFSLAM block-matrix prediction. Only the robot block evolves; landmark blocks are unchanged:

```
P_bar = [ Jfx @ P_RR @ Jfx.T + Jfw @ Qk @ Jfw.T  |  Jfx @ P_RM ]
         [ (Jfx @ P_RM).T                           |  P_MM       ]
```

Accumulated odometry (`rel_disp`, `rel_cov`) is updated using compound pose composition (`Pose3D.oplus`) to track the integrated displacement between graph nodes.

### `Localize(xk_1, Pk_1, k)` — SLAM Loop

```
Prediction
  ↓
GetMeasurements (compass zm)
GetFeatures (bearing+range zf)
DataAssociation (H)
StackMeasurementsAndFeatures → znp (non-paired)
  ↓
If measurement available:
    Update (ISAM2 graph update)
    pose_index += 1
    Reset rel_disp / rel_cov
  ↓
AddNewFeatures (grow state + register in GTSAM)
```

---

## Key Implementation: `EKF.py`

### `Update(...)` — ISAM2 Graph Update

At each pose node, the following factors are added to the ISAM2 graph:

| Factor | GTSAM Type | Purpose |
|--------|-----------|---------|
| Initial anchor | `PriorFactorPose2(X(0))` | Fixes the origin |
| Odometry | `BetweenFactorPose2(X(k), X(k+1))` | Accumulated dead-reckoning |
| Compass | `PoseRotationPrior2D(X(k+1))` | Heading from compass sensor |
| Landmark obs | `BearingRangeFactor2D(X(k+1), L(j))` | Bearing+range to mapped landmarks |

Noise for the odometry factor comes from `rel_cov` — the propagated covariance of accumulated displacement between consecutive graph nodes.

After each ISAM2 update:
- Optimised pose extracted from `result.atPose2(X(pose_idx+1))`
- Optimised landmark positions extracted from `result.atPoint2(L(j))`
- Marginal covariances extracted for both pose and all landmarks via `gtsam.Marginals`
- These replace `self.xk` and `self.Pk`

---

## Data Association

The `DataAssociation()` method (inherited from `FEKFMBL`) matches current observations to known map entries using Mahalanobis distance. Observations that exceed the gate threshold become `znp` (non-paired) and trigger `AddNewFeatures`.

---

## Running

```bash
python3 <main_simulation_script>.py
```

---

## Summary: What's Different in Part 3 vs Earlier Parts

| Aspect | Part 1 | Part 2 | **Part 3** |
|--------|--------|--------|-----------|
| Map at start | None | All known | **None — built on the fly** |
| `nf` at init | 0 | `len(M)` | **0** |
| `AddNewFeatures` active | Yes (EKF only) | No | **Yes (EKF + GTSAM)** |
| Landmark GTSAM nodes | None | Pre-inserted | **Inserted as discovered** |
| `BearingRangeFactor2D` | No | Yes | **Yes** |
| State vector size | Fixed (3) | Fixed (3 + 2N) | **Grows over time** |
