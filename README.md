# untitled_plane_extraction

### Scripts
- **run_benchmark.sh** - runs benchmark on semantic branch
  - **dataset**: kitti, carla (not implemented), both (not implemented)
  - **model-name**: dsnet (others not implemented)
  - **batch-size**
  - **n-steps**
  - **train-size**: percentage, in order from starting scene
  - **scene-size**: number of sampled points from scene
  - **device-name**: string describing train/val device
  - **train**: run training
  - **val**: run validation

### Finding val results
wanb.ai -> skoltech_plane_extraction -> val -> *run_id*

### Prepare:
1. Login on Rucula (the project assumes that you are logged there)
2. `git clone https://github.com/nagonch/untitled_plane_extraction.git`
3. `pip install -r requirements.txt`
4. `bash run_benchmark.sh`
