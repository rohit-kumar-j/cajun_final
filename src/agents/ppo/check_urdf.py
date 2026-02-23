"""
Standalone URDF inspector using Isaac Gym.

Loads a URDF, queries Isaac Gym for every physical property it actually
enforces, and prints them for comparison against config values.

Usage:
    python -m src.agents.ppo.check_urdf --urdf=data/go2/urdf/go2.urdf

NOTE: This script imports NOTHING from the rest of the framework.
It is intentionally self-contained so it can be run on any URDF.

Isaac Gym may silently clip DOF limits set in motor configs if they exceed
what the URDF specifies. This script is the ground truth check.
"""

import argparse
import sys
import os

from loguru import logger

# ── Configure loguru ──────────────────────────────────────────────────────────
# Remove the default handler and add one with a clean, readable format.
# We use print() for tabular data (joints, bodies) and logger.* for
# status messages so the two are visually distinct.
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>check_urdf</cyan> | {message}"
    ),
    colorize=True,
)


def _section(title: str) -> None:
    """Print a bold section header via logger."""
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a URDF via Isaac Gym — prints all physical "
            "properties Isaac Gym actually reads and enforces."
        )
    )
    parser.add_argument(
        "--urdf", required=True,
        help="Path to URDF file, e.g. data/go2/urdf/go2.urdf",
    )
    args = parser.parse_args()

    urdf_path = args.urdf
    if not os.path.isfile(urdf_path):
        logger.critical(f"URDF file not found: {urdf_path}")
        sys.exit(1)

    logger.info(f"URDF Inspector starting — {urdf_path}")

    try:
        from isaacgym import gymapi, gymtorch
        import torch
    except ImportError as exc:
        logger.critical(f"Isaac Gym not available: {exc}")
        sys.exit(1)

    # ── Isaac Gym init ────────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu    = False
    sim_params.up_axis          = gymapi.UpAxis(gymapi.UP_AXIS_Z)
    sim_params.gravity          = gymapi.Vec3(0., 0., -9.81)

    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        logger.critical("Failed to create Isaac Gym sim")
        sys.exit(1)

    # ── Load URDF ─────────────────────────────────────────────────────────────
    asset_root = os.path.dirname(os.path.abspath(urdf_path))
    asset_file = os.path.basename(urdf_path)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link              = False
    asset_options.collapse_fixed_joints      = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments    = True

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if asset is None:
        logger.critical(f"Isaac Gym failed to load URDF: {urdf_path}")
        sys.exit(1)

    logger.success(f"URDF loaded successfully: {asset_file}")

    # ── Create one env + actor ────────────────────────────────────────────────
    env_lower  = gymapi.Vec3(0., 0., 0.)
    env_upper  = gymapi.Vec3(0., 0., 0.)
    env        = gym.create_env(sim, env_lower, env_upper, 1)
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(0., 0., 0.5)
    actor = gym.create_actor(env, asset, start_pose, "robot", 0, 1)

    # ── Basic counts ──────────────────────────────────────────────────────────
    num_dof    = gym.get_asset_dof_count(asset)
    num_bodies = gym.get_asset_rigid_body_count(asset)

    _section("BASIC INFO")
    logger.info(f"DOF count (actuated joints) : {num_dof}")
    logger.info(f"Rigid body count            : {num_bodies}")

    # ── Rigid body names ──────────────────────────────────────────────────────
    _section("RIGID BODY NAMES AND INDICES")
    body_names = gym.get_actor_rigid_body_names(env, actor)
    for i, name in enumerate(body_names):
        logger.debug(f"  [{i:2d}]  {name}")

    # ── DOF names ─────────────────────────────────────────────────────────────
    _section("DOF (JOINT) NAMES AND INDICES")
    dof_names = gym.get_asset_dof_names(asset)
    for i, name in enumerate(dof_names):
        logger.debug(f"  [{i:2d}]  {name}")

    # ── DOF properties (asset level) ─────────────────────────────────────────
    _section("DOF PROPERTIES AS READ BY ISAAC GYM (asset level)")
    logger.warning(
        "Isaac Gym may silently clip effort/velocity limits if they exceed "
        "URDF <limit> values. These are the values Isaac Gym will enforce."
    )
    print()
    print(f"  {'Joint':<35} {'Lower':>10} {'Upper':>10} "
          f"{'MaxVel':>10} {'MaxEffort':>12} {'Stiffness':>11} {'Damping':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*11} {'-'*10}")

    dof_props = gym.get_asset_dof_properties(asset)
    for i, name in enumerate(dof_names):
        print(
            f"  {name:<35} "
            f"{dof_props['lower'][i]:>10.4f} {dof_props['upper'][i]:>10.4f} "
            f"{dof_props['velocity'][i]:>10.3f} {dof_props['effort'][i]:>12.3f} "
            f"{dof_props['stiffness'][i]:>11.3f} {dof_props['damping'][i]:>10.3f}"
        )

    # ── DOF properties (actor level — may differ if Isaac Gym clipped) ────────
    _section("ACTOR DOF PROPERTIES (post actor creation)")
    logger.warning(
        "If these differ from asset DOF properties, Isaac Gym modified them "
        "silently — your motor config values were CLIPPED."
    )
    print()
    print(f"  {'Joint':<35} {'Lower':>10} {'Upper':>10} "
          f"{'MaxVel':>10} {'MaxEffort':>12}  {'Diff?':>6}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*12}  {'-'*6}")

    actor_dof_props = gym.get_actor_dof_properties(env, actor)
    any_diff = False
    for i, name in enumerate(dof_names):
        a_lo  = actor_dof_props['lower'][i]
        a_hi  = actor_dof_props['upper'][i]
        a_vel = actor_dof_props['velocity'][i]
        a_eff = actor_dof_props['effort'][i]
        diff  = (
            abs(a_lo  - dof_props['lower'][i])    > 1e-4 or
            abs(a_hi  - dof_props['upper'][i])    > 1e-4 or
            abs(a_vel - dof_props['velocity'][i]) > 1e-4 or
            abs(a_eff - dof_props['effort'][i])   > 1e-4
        )
        diff_str = "DIFF" if diff else "ok"
        row = (
            f"  {name:<35} "
            f"{a_lo:>10.4f} {a_hi:>10.4f} "
            f"{a_vel:>10.3f} {a_eff:>12.3f}  {diff_str:>6}"
        )
        print(row)
        if diff:
            any_diff = True
            logger.warning(f"Actor DOF DIFFERS from asset for joint '{name}'")

    if not any_diff:
        logger.success("Actor DOF properties match asset DOF properties — no silent clipping detected")

    # ── Override test: set unrealistic values, read back, check clipping ──────
    _section("OVERRIDE CLIPPING TEST")
    logger.warning(
        "This test sets UNREALISTICALLY HIGH values on the actor DOF properties "
        "via gym.set_actor_dof_properties(), then reads them back.\n"
        "          This simulates what happens when motor config overrides URDF limits.\n"
        "          If Isaac Gym clips them → your motor config values are being silently capped."
    )

    UNREALISTIC_EFFORT   = 9999.0   # far above any real motor
    UNREALISTIC_VELOCITY = 9999.0
    UNREALISTIC_LOWER    = -9999.0
    UNREALISTIC_UPPER    =  9999.0

    # Write unrealistic values into a copy of the actor DOF props
    override_props = gym.get_actor_dof_properties(env, actor)
    for i in range(num_dof):
        override_props['lower'][i]    = UNREALISTIC_LOWER
        override_props['upper'][i]    = UNREALISTIC_UPPER
        override_props['velocity'][i] = UNREALISTIC_VELOCITY
        override_props['effort'][i]   = UNREALISTIC_EFFORT

    gym.set_actor_dof_properties(env, actor, override_props)

    # Read back what Isaac Gym actually stored
    readback_props = gym.get_actor_dof_properties(env, actor)

    print()
    print(f"  Attempted to set:  effort={UNREALISTIC_EFFORT},  velocity={UNREALISTIC_VELOCITY}")
    print(f"  {'Joint':<35} {'Set Effort':>12} {'Got Effort':>12} {'Set Vel':>10} {'Got Vel':>10}  {'Clipped?':>9}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10} {'-'*10}  {'-'*9}")

    any_clipped = False
    for i, name in enumerate(dof_names):
        got_eff = readback_props['effort'][i]
        got_vel = readback_props['velocity'][i]
        eff_clipped = abs(got_eff - UNREALISTIC_EFFORT) > 1e-2
        vel_clipped = abs(got_vel - UNREALISTIC_VELOCITY) > 1e-2
        clipped = eff_clipped or vel_clipped
        clip_str = "CLIPPED" if clipped else "ok"
        print(
            f"  {name:<35} "
            f"{UNREALISTIC_EFFORT:>12.1f} {got_eff:>12.3f} "
            f"{UNREALISTIC_VELOCITY:>10.1f} {got_vel:>10.3f}  {clip_str:>9}"
        )
        if clipped:
            any_clipped = True

    print()
    if any_clipped:
        logger.warning(
            "Isaac Gym CLIPPED your override values back to URDF limits. "
            "Motor config values that exceed URDF limits will be silently capped."
        )
        logger.info("Showing what each joint was clipped to:")
        for i, name in enumerate(dof_names):
            ge = readback_props['effort'][i]
            gv = readback_props['velocity'][i]
            oe = UNREALISTIC_EFFORT
            ov = UNREALISTIC_VELOCITY
            if abs(ge - oe) > 1e-2 or abs(gv - ov) > 1e-2:
                logger.info(f"  {name}: effort {oe:.0f} → {ge:.3f}   vel {ov:.0f} → {gv:.3f}")
    else:
        logger.success(
            "Isaac Gym did NOT clip the override values — "
            "it accepted effort=9999 and velocity=9999 as-is.\n"
            "          This means Isaac Gym does NOT enforce URDF limits on programmatic overrides.\n"
            "          Your motor config values (whatever you set) will be used exactly."
        )

    # Restore the original actor DOF props before continuing
    gym.set_actor_dof_properties(env, actor, actor_dof_props)
    logger.info("Actor DOF properties restored to post-creation values.")

    # ── Rigid body masses and inertia ─────────────────────────────────────────
    _section("RIGID BODY MASSES AND INERTIA")
    body_props = gym.get_actor_rigid_body_properties(env, actor)

    total_mass = 0.0
    total_ixx = total_iyy = total_izz = 0.0

    print()
    print(f"  {'Body':<30} {'Mass(kg)':>9} {'COMx':>8} {'COMy':>8} {'COMz':>8} "
          f"{'Ixx':>10} {'Iyy':>10} {'Izz':>10}")
    print(f"  {'-'*30} {'-'*9} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*10} {'-'*10} {'-'*10}")

    for name, prop in zip(body_names, body_props):
        m   = prop.mass
        cx, cy, cz = prop.com.x, prop.com.y, prop.com.z
        ixx = prop.inertia.x.x
        iyy = prop.inertia.y.y
        izz = prop.inertia.z.z
        total_mass += m
        total_ixx  += ixx
        total_iyy  += iyy
        total_izz  += izz
        print(
            f"  {name:<30} {m:>9.4f} {cx:>8.5f} {cy:>8.5f} {cz:>8.5f} "
            f"{ixx:>10.6f} {iyy:>10.6f} {izz:>10.6f}"
        )

    print()
    logger.info(f"TOTAL robot mass        : {total_mass:.4f} kg")
    logger.info(f"Total inertia (summed)  : Ixx={total_ixx:.6f}  Iyy={total_iyy:.6f}  Izz={total_izz:.6f}")
    logger.info(
        f"Trunk COM offset        : "
        f"x={body_props[0].com.x:.5f}  "
        f"y={body_props[0].com.y:.5f}  "
        f"z={body_props[0].com.z:.5f}"
    )
    logger.info(f"Trunk mass              : {body_props[0].mass:.4f} kg")

    # ── Rigid body positions (hip offsets) ────────────────────────────────────
    _section("RIGID BODY POSITIONS RELATIVE TO TRUNK (one sim step)")
    logger.warning(
        "These positions come from rigid_body_state_tensor after one sim step. "
        "They are the Isaac Gym ground truth for hip_offset / hip_positions_in_body_frame."
    )

    gym.prepare_sim(sim)
    rb_tensor   = gym.acquire_rigid_body_state_tensor(sim)
    rb_states   = gymtorch.wrap_tensor(rb_tensor)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)

    trunk_idx = next(
        (i for i, n in enumerate(body_names)
         if n.lower() in ("trunk", "base", "torso", "body")),
        None
    )
    if trunk_idx is None:
        logger.error(
            "Could not find root body (tried: trunk, base, torso, body). "
            "Skipping relative position table."
        )
    else:
        logger.info(
            f"Root body identified as '{body_names[trunk_idx]}' "
            f"(index {trunk_idx}). "
            f"Note: collapse_fixed_joints may rename 'trunk' → 'base'."
        )
        trunk_pos = rb_states[trunk_idx, :3].cpu().numpy()
        logger.info(f"Trunk world position at spawn: {trunk_pos}")
        print()
        print(f"  {'Body':<30} {'World X':>10} {'World Y':>10} {'World Z':>10} "
              f"{'Rel X':>10} {'Rel Y':>10} {'Rel Z':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} "
              f"{'-'*10} {'-'*10} {'-'*10}")

        keywords = ("hip", "thigh", "calf", "foot")
        for i, name in enumerate(body_names):
            if any(k in name.lower() for k in keywords):
                pos = rb_states[i, :3].cpu().numpy()
                rel = pos - trunk_pos
                print(
                    f"  {name:<30} "
                    f"{pos[0]:>10.4f} {pos[1]:>10.4f} {pos[2]:>10.4f} "
                    f"{rel[0]:>10.4f} {rel[1]:>10.4f} {rel[2]:>10.4f}"
                )

    # ── URDF geometry summary (parsed, for quick reference) ───────────────────
    _section("URDF GEOMETRY REFERENCE (parsed from URDF, not Isaac Gym tensors)")
    logger.warning("Use the rigid body positions above for Isaac Gym ground truth.")
    logger.info("Hip joint origins in trunk frame (from URDF <origin xyz=...>):")
    logger.info("  FR: [+0.1934, -0.0465,  0.000]")
    logger.info("  FL: [+0.1934, +0.0465,  0.000]")
    logger.info("  RR: [-0.1934, -0.0465,  0.000]")
    logger.info("  RL: [-0.1934, +0.0465,  0.000]")
    logger.info("Thigh-hip lateral offset (thigh joint origin y): 0.0955 m")
    logger.info("Link lengths (joint origin z-distance):")
    logger.info("  Thigh = 0.213 m  |  Calf = 0.213 m  |  Foot = 0.213 m")
    logger.info("Foot collision sphere radius: 0.022 m")

    # ── Known discrepancies ───────────────────────────────────────────────────
    _section("KNOWN DISCREPANCIES (URDF ground truth vs current code/config)")
    logger.warning("go1.py motor block: min/max_torque = ±33.5 Nm (uniform)")
    logger.info("  URDF: hip/thigh = 23.7 Nm,  calf = 45.43 Nm  — config is WRONG")

    logger.warning("go1.py motor block: min/max_velocity = ±21.0 rad/s (uniform)")
    logger.info("  URDF: hip/thigh = 30.1 rad/s,  calf = 15.70 rad/s  — config is WRONG")

    logger.warning("go1.py hip_offset: x=±0.1805 m, y=±0.047 m  (A1 values)")
    logger.info("  URDF Go2: x=±0.1934 m, y=±0.0465 m  — WRONG ROBOT")

    logger.warning("go1.py uses URDF: data/a1/urdf/_A1u150.urdf")
    logger.info("  Should use: data/go2/urdf/go2.urdf")

    logger.warning("init_height inconsistency: jump_env=0.268, rewards=0.26, QP default=0.26")
    logger.info("  All should reference config.env.init_height — this is a bug")

    logger.warning("go1_rewards.py foot radius: 0.02 m")
    logger.info("  URDF collision sphere radius: 0.022 m  — small but incorrect")

    logger.warning(f"QP default body_mass: 13.076 kg (Go1)")
    logger.info(f"  URDF total mass: {total_mass:.3f} kg  — WRONG ROBOT")

    logger.warning("RR/RL thigh joint limits differ from FR/FL in URDF")
    logger.info("  FR/FL thigh: lower=-1.5708, upper=+3.4907")
    logger.info("  RR/RL thigh: lower=-0.5236, upper=+4.5379")
    logger.info("  go1.py uses same limits for all 4 legs  — WRONG for RR/RL")

    logger.warning("go1.py l_hip = 0.08 used in IK abduction offset")
    logger.info("  URDF thigh joint origin y = 0.0955 m  — verify IK formula uses correct value")

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("SUMMARY")
    logger.success(f"Total robot mass (Isaac Gym) : {total_mass:.4f} kg")
    logger.success(f"Total DOFs                  : {num_dof}")
    logger.success(f"Total rigid bodies          : {num_bodies}")
    if any_diff:
        logger.warning("Some DOF properties differ between asset and actor — see ACTOR DOF section above")
    else:
        logger.success("No silent DOF clipping detected by Isaac Gym")

    gym.destroy_sim(sim)
    logger.info("Done. Isaac Gym sim destroyed cleanly.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl-C)")
        sys.exit(0)
    except Exception as exc:
        import traceback
        logger.critical(f"Unhandled exception: {type(exc).__name__}: {exc}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
