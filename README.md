Retain Prototype Processing:

python build_retain_3branch.py \
  --zh-dir ZH_part \
  --probe-source 你的固定source音频.mp3 \
  --n-speakers 50 \
  --output-dir ./retain_3branch_out



Forget Audio Processing:

1. Generate cache for forget audio
python inference_v2.py \
  --source probe.wav \
  --target ZH_part/ZH_B00000_S09990_W000000.mp3 \
  --output ./output \
  --diffusion-steps 30 \
  --convert-style false \
  --anonymization-only false \
  --trus-enable true \
  --trus-mode record \
  --trus-cache-out ./artifacts/forget999_cache.pt

2. Cache to summary
python cache_to_summary.py \
  --cache ./artifacts/forget999_cache.pt \
  --out ./artifacts/forget999_summary.pt

3. Make forget profile 
python build_forget_profile.py \
  --prototype ./retain_bulk_out/prototype.pt \
  --forget-summary ./artifacts/forget976_summary.pt \
  --out ./artifacts/forget976_profile_3branch.pt \
  --k 1.0

4. Steer
python inference_v2.py \
  --source your_source.wav \
  --target your_target.wav \
  --output ./output \
  --diffusion-steps 30 \
  --convert-style false \
  --anonymization-only false \
  --trus-enable true \
  --trus-mode steer \
  --trus-profile ./artifacts/forget976_profile_3branch.pt \
  --trus-alpha-branch0 0.4 \
  --trus-alpha-branch1 0.1 \
  --trus-alpha-branch2 0.0