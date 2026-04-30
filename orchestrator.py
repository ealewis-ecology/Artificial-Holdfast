"""Pipeline orchestrator: regen d3/d6/d7 STLs via haptera_export.py, repair
via headless Blender, verify watertightness, then run fractal_only.py at
pitch=1.0 mm.

Designed to run autonomously after launch. Writes:
  PIPELINE_STATUS.txt  -- chronological log of each stage
  FINAL_RESULTS.txt    -- aggregated fractal output (or failure summary)

Exit codes:
  0 -- success: fractal results written, shutdown scheduled
  1 -- haptera_export.py errored on >=1 depth
  2 -- repaired STLs not all watertight (after Blender repair)
  3 -- fractal_only.py errored on >=1 depth
"""

import os
import sys
import time
import subprocess
from pathlib import Path

HOME = Path("/home/ubuntu/Artificial-Holdfast")
PY = HOME / ".venv" / "bin" / "python"
BLENDER = "/usr/bin/blender"
PITCH_MM = 1.0
DEPTHS = [3, 6, 7]
SHUTDOWN_DELAY_MIN = 120
REPAIR_VOXEL_MM = 0.5

STATUS = HOME / "PIPELINE_STATUS.txt"
RESULTS = HOME / "FINAL_RESULTS.txt"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def stl_raw(d):
    return HOME / f"haptera_d{d}_k2_r130_h130_f650.stl"


def stl_repaired(d):
    return HOME / f"haptera_d{d}_k2_r130_h130_f650_repaired.stl"


def export_log(d):
    return HOME / f"export_d{d}.log"


def repair_log(d):
    return HOME / f"repair_d{d}.log"


def fractal_log(d):
    return HOME / f"frac_d{d}.log"


def run_parallel(label, builder):
    """Launch one subprocess per depth in parallel and wait for all."""
    procs = {}
    for d in DEPTHS:
        cmd, env, out_path = builder(d)
        log(f"  launching {label} d={d}  -> {out_path.name}")
        procs[d] = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(HOME),
            stdout=open(out_path, "w"),
            stderr=subprocess.STDOUT,
        )
    rcs = {}
    for d, p in procs.items():
        rc = p.wait()
        rcs[d] = rc
        log(f"  {label} d={d} exited (code={rc})")
    return rcs


def stage_export():
    log("=== Stage 1: Regenerate STLs via haptera_export.py (parallel) ===")

    def builder(d):
        env = os.environ.copy()
        env["HAPTERA_DEPTH"] = str(d)
        env["HAPTERA_OUTPUT_DIR"] = str(HOME)
        return ([str(PY), str(HOME / "haptera_export.py")],
                env, export_log(d))

    rcs = run_parallel("export", builder)
    return all(rc == 0 for rc in rcs.values())


def stage_repair():
    log(f"=== Stage 2: Repair via headless Blender "
        f"(voxel fallback @ {REPAIR_VOXEL_MM} mm) ===")

    def builder(d):
        return ([BLENDER, "--background", "--python",
                 str(HOME / "blender_repair.py"), "--",
                 str(stl_raw(d)), str(stl_repaired(d)), str(REPAIR_VOXEL_MM)],
                os.environ.copy(),
                repair_log(d))

    rcs = run_parallel("repair", builder)
    # Blender repair script exits 0 if final mesh is watertight (manifold),
    # 1 if non-manifold edges remain after fallback. Treat both as
    # "completed" — we still re-verify with trimesh below.
    for d, rc in rcs.items():
        log(f"  repair d={d}: exit={rc}  "
            f"({'watertight' if rc == 0 else 'NON-MANIFOLD remaining'})")
    return rcs


def stage_check_watertight():
    log("=== Stage 3: Verify watertightness via trimesh on repaired STLs ===")
    sys.path.insert(0, str(HOME))
    import trimesh

    statuses = {}
    for d in DEPTHS:
        sp = stl_repaired(d)
        if not sp.exists():
            log(f"  d={d}: MISSING repaired STL ({sp.name})")
            statuses[d] = False
            continue
        size = sp.stat().st_size
        m = trimesh.load(str(sp), force="mesh")
        wt = bool(m.is_watertight)
        log(f"  d={d}: watertight={wt}  verts={len(m.vertices):,}  "
            f"faces={len(m.faces):,}  size={size:,}B")
        statuses[d] = wt
    return statuses


def stage_fractal():
    log(f"=== Stage 4: Box-counting fractal dimension at pitch={PITCH_MM} mm "
        f"(parallel) ===")

    def builder(d):
        return ([str(PY), str(HOME / "fractal_only.py"),
                 str(stl_repaired(d)), str(PITCH_MM)],
                os.environ.copy(),
                fractal_log(d))

    rcs = run_parallel("fractal", builder)
    return all(rc == 0 for rc in rcs.values())


def write_results(watertight_statuses, fractal_ran):
    log("=== Stage 5: Write FINAL_RESULTS.txt ===")
    with open(RESULTS, "w") as f:
        f.write("=== FRACTAL DIMENSION COMPARISON ===\n")
        f.write(f"pitch        : {PITCH_MM} mm  (1.0 mm edge length)\n")
        f.write(f"voxel volume : {PITCH_MM ** 3:.6f} mm^3\n")
        f.write(f"depths       : {DEPTHS}\n")
        f.write(f"repair tool  : Blender voxel-remesh fallback @ "
                f"{REPAIR_VOXEL_MM} mm\n\n")

        for d in DEPTHS:
            f.write("\n" + "=" * 78 + "\n")
            f.write(f"d = {d}   (haptera_d{d}_k2_r130_h130_f650_repaired.stl)\n")
            f.write(f"watertight = {watertight_statuses.get(d)}\n")
            f.write("=" * 78 + "\n")
            fp = fractal_log(d)
            if fp.exists():
                f.write(fp.read_text())
            else:
                f.write("(no fractal log present)\n")
            f.write("\n")


def stage_shutdown():
    log(f"=== Stage 6: Schedule shutdown +{SHUTDOWN_DELAY_MIN} min ===")
    rc = subprocess.run(
        ["sudo", "-n", "shutdown", "-h", f"+{SHUTDOWN_DELAY_MIN}"],
        capture_output=True, text=True,
    )
    log(f"  shutdown returncode: {rc.returncode}")
    if rc.stdout:
        log(f"  stdout: {rc.stdout.strip()}")
    if rc.stderr:
        log(f"  stderr: {rc.stderr.strip()}")
    return rc.returncode == 0


def main():
    STATUS.write_text("")
    log("=== Pipeline start ===")
    log(f"  DEPTHS={DEPTHS}  PITCH={PITCH_MM}mm  REPAIR_VOXEL={REPAIR_VOXEL_MM}mm")

    if not stage_export():
        log("EXPORT FAILED on >=1 depth — see export_d*.log")
        with open(RESULTS, "w") as f:
            f.write("FAILED at Stage 1 (haptera_export.py).\n")
            f.write("See export_d*.log for details.\n")
            f.write("EC2 instance left running for debugging.\n")
        sys.exit(1)

    stage_repair()  # don't bail on rc — verify next stage with trimesh

    statuses = stage_check_watertight()
    if not all(statuses.values()):
        log("Repaired STLs are NOT all watertight — stopping (Blender repair "
            "could not produce manifold output)")
        with open(RESULTS, "w") as f:
            f.write("STOPPED at Stage 3: repaired STLs not all watertight.\n")
            f.write("Blender repair could not produce manifold output.\n")
            f.write("EC2 instance left running for debugging.\n\n")
            for d in DEPTHS:
                f.write(f"  d={d}: watertight={statuses[d]}\n")
        sys.exit(2)

    fractal_ok = stage_fractal()
    write_results(statuses, fractal_ok)

    if not fractal_ok:
        log("FRACTAL FAILED on >=1 depth — see frac_d*.log; not shutting down")
        sys.exit(3)

    stage_shutdown()
    log("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
