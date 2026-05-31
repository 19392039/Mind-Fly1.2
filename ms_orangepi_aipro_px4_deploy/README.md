# MindSpore PPO Orange Pi AIpro PX4 Deploy

This folder is independent from the existing project files. It adapts the local
`mind_infer.py` MindSpore PPO policy to the torch deployment style:

- RPLidar reads 128 rays in a background thread.
- PX4 is controlled with MAVLink body-frame `vx` and `yaw_rate`.
- The policy input is the real MindSpore 135-dim observation.
- The policy output is `[vx_mps, omega_radps]`, not torch's `[vx, vy]`.

## Files

```text
real_flight_ms.py   main loop for Orange Pi AIpro + RPLidar + PX4
flight_control.py   MAVLink PX4 velocity/yaw-rate controller
lidar_reader.py     RPLidar 360-degree scan to 128-ray adapter
runtime_common.py   model loading, observation building, safety filter
deploy_policy.py    deterministic actor wrapper
export_mindir.py    ckpt to MindIR
run_mindir_policy.py MindIR smoke test
convert_om.sh       MindIR to OM template for CANN atc
requirements.txt    Python-side packages except MindSpore itself
```

## Observation And Action

Observation shape:

```text
[1, 135]
[128 normalized lidar rays,
 ref_sin, ref_cos,
 prev_vx_n, prev_omega_n,
 dvx_n, domega_n,
 dist_n]
```

Action shape:

```text
[1, 2] = [vx_mps, omega_radps]
```

## Install On Orange Pi AIpro

Copy the whole project to the board, then:

```bash
cd ~/Mind-Fly_1.2
python3 -m venv ~/mindfly_ms_env
source ~/mindfly_ms_env/bin/activate
pip install -U pip
pip install -r ms_orangepi_aipro_px4_deploy/requirements.txt
```

Install the MindSpore package that matches the board image and CANN version.
Then run a no-hardware smoke test:

```bash
python3 ms_orangepi_aipro_px4_deploy/real_flight_ms.py --dry-run
```

## Run With Hardware

Default ports follow the torch reference:

```bash
python3 ms_orangepi_aipro_px4_deploy/real_flight_ms.py \
  --device-target CPU \
  --lidar-port /dev/ttyUSB0 \
  --px4-port /dev/ttyAMA1 \
  --target-x 10.0 \
  --target-y 0.0
```

If your MindSpore installation supports Ascend execution:

```bash
python3 ms_orangepi_aipro_px4_deploy/real_flight_ms.py \
  --device-target Ascend \
  --graph-mode
```

## Benchmark Inference Speed

Model-only inference latency:

```bash
python3 ms_orangepi_aipro_px4_deploy/benchmark_infer.py \
  --device-target CPU \
  --warmup 50 \
  --iters 500
```

Dry control loop latency, including lidar array preparation, observation
building, inference, and safety filtering:

```bash
python3 ms_orangepi_aipro_px4_deploy/benchmark_infer.py \
  --device-target CPU \
  --mode dry-loop \
  --warmup 50 \
  --iters 500
```

The script reports mean, p50, p95, p99, max latency, and `FPS / Hz`.
Use p95 and max latency to judge stability, not only average FPS.

## Lidar Alignment

The model assumes:

```text
ray 0   forward
ray 32  left
ray 64  back
ray 96  right
```

The default `--lidar-angle-offset 180` follows the provided torch deployment
reader. If the front sector is wrong, adjust it in 90-degree steps first:

```bash
--lidar-angle-offset 0
--lidar-angle-offset 90
--lidar-angle-offset 180
--lidar-angle-offset 270
```

## Export

MindIR:

```bash
python3 ms_orangepi_aipro_px4_deploy/export_mindir.py \
  --output ms_orangepi_aipro_px4_deploy/artifacts/ppo_policy
```

OM template:

```bash
SOC_VERSION=Ascend310B1 bash ms_orangepi_aipro_px4_deploy/convert_om.sh
```

Use the `SOC_VERSION` required by your Orange Pi AIpro CANN image.
