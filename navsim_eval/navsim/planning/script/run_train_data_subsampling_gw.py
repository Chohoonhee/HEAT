import os
import random
import shutil

# ✅ 시드 고정
seed = 2024
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# 원본 디렉토리
# NOTE: with_heading ############
src_dir = "navsim_cache_out/pkl_extracted_heading/train"
# dst_dir = "navsim_cache_out/pkl_extracted_heading/train_subsampled"         # sample_ratio = 0.33
# dst_dir = "navsim_cache_out/pkl_extracted_heading/train_subsampled_9_5k"    # sample_ratio = 0.112
#################################

# NOTE: without_heading #########
# src_dir = "navsim_cache_out/pkl_extracted/train"
# dst_dir = "navsim_cache_out/pkl_extracted/train_subsampled"               # sample_ratio = 0.33
dst_dir = "navsim_cache_out/pkl_extracted/train_subsampled_9_5k"          # sample_ratio = 0.112
#################################

# 샘플링 비율 (예: 30%)
sample_ratio = 0.112

# 전체 pkl 파일 목록 (정렬 → 순서 고정)
all_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".pkl")])

# 랜덤 샘플링
sample_size = int(len(all_files) * sample_ratio)
sampled_files = random.sample(all_files, sample_size)

# 저장 폴더 생성
os.makedirs(dst_dir, exist_ok=True)

# 선택된 파일 복사
for f in sampled_files:
    shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

print(f"총 {len(all_files)}개 중 {len(sampled_files)}개를 샘플링하여 {dst_dir}에 저장했습니다.")
