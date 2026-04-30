# EC2 Instance — Handoff

The EC2 instance is a long-lived workhorse for two compute-heavy jobs:

1. Generating large haptera STL meshes via `haptera_export.py` (uses
   manifold3d boolean unions of tens of thousands of cylinder primitives).
2. Slicing those meshes with BambuStudio's CLI to produce printer gcode.

Both run on the persistent EBS root volume, so a stop/start cycle preserves
all installed software and cached files. **A terminate would destroy
everything** — see "Shutdown vs terminate" below.

## EC2 access

- IP: `18.222.189.2` (changes on every stop/start — ask the user for the new IP each time)
- User: `ubuntu`
- Private key (laptop): `/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/Eli's MacBook.pem`
- Region: us-east-2 (Ohio)
- Instance type: `c6a.32xlarge` — 128 vCPU, 246 GiB RAM, x86_64, Ubuntu 24.04
- On-demand price: **$4.896/hour** — always shut down when work is done
- Root EBS: 48 GB (~9 GB used at last check; the haptera STLs and 3mfs are the bulk)

Quick test:
```bash
ssh -i "/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/Eli's MacBook.pem" \
  ubuntu@<ip> "uptime && nproc && free -g | head -2"
```

## What's installed on the persistent disk

| Path | What it is |
| ---- | ---- |
| `~/bambu/BambuStudio.AppImage` | v02.06.00.51, ubuntu-24.04 build |
| `~/bambu/squashfs-root/` | Extracted AppImage contents |
| `~/bambu/squashfs-root/AppRun` | Wrapper that sets `LD_LIBRARY_PATH` etc. before launching the CLI |
| `~/bambu/squashfs-root/bin/bambu-studio` | Raw binary — **do not call directly**, it can't find `libavcodec.so.61` etc. without AppRun's env |
| `~/Artificial-Holdfast/.venv/` | Python 3.12 venv with `trimesh`, `manifold3d`, `numpy` for `haptera_export.py` |
| `~/Artificial-Holdfast/haptera_export.py` | The mesh generator (synced from the laptop git repo) |
| `~/Artificial-Holdfast/monitor.sh` | Status emitter for long-running slices/exports |

Apt packages installed for the AppImage runtime:
```
libgl1 libglu1-mesa libgtk-3-0 libwebkit2gtk-4.1-0 libsoup-3.0-0
xvfb libegl1 libosmesa6
```

If you ever need to reinstall on a fresh disk (terminate + relaunch, etc.):
```bash
mkdir -p ~/bambu && cd ~/bambu
curl -L -o BambuStudio.AppImage \
  'https://github.com/bambulab/BambuStudio/releases/download/v02.06.00.51/BambuStudio_ubuntu-24.04-v02.06.00.51-20260417160415.AppImage'
chmod +x BambuStudio.AppImage
./BambuStudio.AppImage --appimage-extract
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  libgl1 libglu1-mesa libgtk-3-0 libwebkit2gtk-4.1-0 libsoup-3.0-0 \
  xvfb libegl1 libosmesa6
# Python venv
sudo apt-get install -y python3.12-venv
python3 -m venv ~/Artificial-Holdfast/.venv
~/Artificial-Holdfast/.venv/bin/pip install trimesh manifold3d numpy
```

## Workflow A: generate a haptera STL

```bash
ssh -i "<key>" ubuntu@<ip>
cd ~/Artificial-Holdfast
# Edit DEPTH (line 26 of haptera_export.py) for the resolution you want
sed -i 's/^DEPTH  = 9 #Number of nodes/DEPTH  = 7 #Number of nodes/' haptera_export.py

# Launch detached so SSH disconnect won't kill it
date -u +%s > slice.start_epoch  # used by monitor.sh
setsid nohup bash -c 'source .venv/bin/activate && python3 -u haptera_export.py' \
  > run_export.log 2>&1 < /dev/null &
echo $! > run_export.pid
```

Per `haptera_export.py`'s output naming, the result lands at
`~/Artificial-Holdfast/haptera_d{DEPTH}_k2_r130_h130_f650.{stl,txt,cache.txt}`.

Approximate runtimes & memory (on this c6a.32xlarge):

| DEPTH | segments | RSS peak | wall time | STL size |
| ----- | -------- | -------- | --------- | -------- |
| 6 | ~5K | ~5 GB | ~1 min | ~92 MB |
| 7 | ~10K | ~12 GB | ~3 min | ~207 MB |
| 9 | ~39K | ~40+ GB | ~30 min | ~1.18 GB |

`haptera_d{DEPTH}_*.cache.txt` records the converged radius multiplier so
re-runs at the same DEPTH start from a converged state and finish in 1
iteration. Don't delete the cache files unless you change geometry params.

## Workflow B: slice a 3mf with BambuStudio CLI

The CLI binary needs an X display, so wrap with `xvfb-run` and call `AppRun`
(not `bin/bambu-studio` directly):

```bash
cd ~/Artificial-Holdfast
rm -rf gcode_out && mkdir gcode_out
rm -f slice.log slice.pid
date -u +%s > slice.start_epoch

setsid nohup bash -c 'cd /home/ubuntu/bambu/squashfs-root && \
  xvfb-run -a ./AppRun --debug 4 --slice 0 \
  --outputdir /home/ubuntu/Artificial-Holdfast/gcode_out \
  /home/ubuntu/Artificial-Holdfast/<input>.3mf' \
  > slice.log 2>&1 < /dev/null &

# The bash backgrounding gives you the wrapping shell PID, not the slicer's.
# After ~3 s, find the actual bambu-studio PID and write it for monitor.sh:
sleep 3
ps -ef | grep -E 'AppRun|bambu-studio' | grep -v grep
echo <bambu-studio-pid> > slice.pid
```

Notes on the command:
- `--slice 0` slices all plates (`1`, `2`, … = specific plate index)
- `--debug 4` adds info-level logging to stdout
- Settings priority (per `--help`): CLI flags > `--load-settings` / `--load-filaments` > settings inside the 3mf
- To honor the preset embedded in the 3mf, pass **no** `--load-settings`
- The `setsid` is important — without it, even a `nohup`-wrapped process can be torn down when the SSH parent exits in some shells

The log emits `default_status_callback: percent=X` lines at major phase
boundaries:
```
5  = Slicing mesh
15 = Generating walls
25 = Generating infill regions
35 = Generating infill toolpath
50 = Generating support
80 = Generating G-code: layer N
```
Between those, expect long silent CPU-bound stretches.

## Workflow C: turn the gcode into a print-ready .gcode.3mf

Once slicing finishes, the printer wants a single `.gcode.3mf` with the
gcode embedded — not a loose `.gcode` file alongside a project `.3mf`.

The merge happens locally on the laptop (no EC2 needed), via:
`/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/generate-bbl/embed-gcode-into-3mf.sh`

See `generate-bbl/HANDOFF.md` for the full why/how. Quick form:
```bash
cd "/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/generate-bbl"
./embed-gcode-into-3mf.sh \
  ../models/<name>.3mf \
  ../models/<name>.gcode \
  ../models/<name>.gcode.3mf
```
Then copy `<name>.gcode.3mf` (and the loose `.gcode` if you also want it
direct on the SD card) to `/Volumes/NO NAME/cache/`.

**Known bug, fixed 2026-04-29:** the script's `header_field` calls used
single-escaped brackets (`\[g\]`) which silently failed to match in awk's
dynamic regex, producing `weight=0g` in the output. The repo copy now uses
double-escaped brackets (`\\[g\\]`). If you copy the script anywhere else,
keep the double escape.

## Monitor script

`~/Artificial-Holdfast/monitor.sh` polls both `slice.pid` and
`run_export.pid` on a configurable interval and emits one status line per
tick (cost, RSS, percent, last log message). It also detects completion
and announces the gcode file path / size / watertight status.

```bash
INTERVAL=300 ./monitor.sh
```

Within Claude sessions it's easier to drive the same checks from a local
`Monitor` task that ssh's in every 2 min and prints only on
percent-change. See prior session transcripts for the exact wrapper.

## File transfer

```bash
# Upload a 3mf
scp -i "<key>" "<local 3mf>" ubuntu@<ip>:/home/ubuntu/Artificial-Holdfast/

# Download a sliced gcode
scp -i "<key>" ubuntu@<ip>:/home/ubuntu/Artificial-Holdfast/gcode_out/*.gcode \
  "/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/models/<name>.gcode"

# Download an exported STL
scp -i "<key>" ubuntu@<ip>:/home/ubuntu/Artificial-Holdfast/haptera_d{N}_*.{stl,txt} \
  "/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/models/"
```

## Shutdown vs terminate

| Action | Command | Cost when idle | Disk preserved? |
| ------ | ------- | -------------- | --------------- |
| Stop (recommended) | `ssh ... "sudo shutdown -h now"` | ~$0.10/mo for the EBS volume | **Yes** |
| Terminate (destructive) | AWS console / `aws ec2 terminate-instances` | $0 | **No — wipes BambuStudio install, .venv, all STLs/3mfs** |

Always prefer stop. The disk-resident installs (BambuStudio, the python venv
with trimesh+manifold3d, the cached `.cache.txt` convergence files) take
30+ min to recreate from scratch; preserving the EBS volume is essentially
free.

## Known issue: support generation can stall on dense haptera models

A previous attempt to slice `haptera_d9_k2_r130_h130_f650.3mf` (1,720 parts,
7.7 M facets, organic tree supports) reached **50% "Generating support"**
and ran for **8 h+** without progressing past that phase before being
killed. Memory peaked around 197 GB. Active CPU dropped from ~70 cores to
~7 in the support phase.

Mitigations to try **before** re-slicing a dense model:
- Reduce `support_object_xy_distance` / disable tree-organic supports (use
  normal grid supports), or
- Print without supports — the d7/d9 STL exports use `ENDPOINTS_TO_FLOOR=True`
  in [haptera_export.py](haptera_export.py:27) so every leaf branch reaches
  z=0 and there are no overhangs to support, or
- Reduce `DEPTH` (e.g. 6 or 7) to cut the part count

If the slice plateaus at 50% for >1 h with the percent counter not
advancing, plan to kill it and revisit support settings in BambuStudio
before re-uploading the 3mf.

## File-naming convention

Generated filenames encode the geometry parameters so two 3mfs at different
DEPTHs / radii / heights / fill-fractions don't collide:

```
haptera_d{DEPTH}_k{K}_r{CONE_R}_h{CONE_H}_f{FILL_FRAC*1000}[.suffix].{stl|3mf|gcode|gcode.3mf|cache.txt|txt}
```

Examples on the disk right now:
```
haptera_d6_k2_r130_h130_f650.{stl,txt,cache.txt}            # ~92 MB STL
haptera_d6_k2_r130_h130_f650_4mm.3mf                        # 20 MB project 3mf (currently slicing)
haptera_d7_k2_r130_h130_f650.{stl,3mf,txt,cache.txt}        # 207 MB STL, 62 MB project 3mf
haptera_d9_k2_r130_h130_f650.{stl,3mf,txt,cache.txt}        # 1.18 GB STL, 27 MB project 3mf
haptera_d9_k2_r130_h130_f650_A1.3mf                         # variant for the A1 printer
```

## State summary at handoff (2026-04-29)

- Instance `c6a.32xlarge`, IP `18.222.189.2`, currently **running**
- Slice in progress: `haptera_d6_k2_r130_h130_f650_4mm.3mf` — at 50%
  (Generating support) for ~25 min as of last check, watching for plateau
- Persistent disk has the d6, d7, d9 STL/3mf/cache files (see file listing
  above) and the BambuStudio install
- Recent prior slices: d7 sliced cleanly in ~11 min, produced 813.9 MB
  gcode; d6_4mm is the next target
- Local print-ready outputs from this batch live in
  `/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/models/` and on
  the printer SD card at `/Volumes/NO NAME/cache/`
