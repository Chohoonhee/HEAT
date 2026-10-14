import os
import re
import random
from math import floor

data_path = './waymo_e2e_training'
text_path = './train_sample_28100_stride2.txt'
target_total = 28100
seed = 42
STRIDE = 2  # ← 번호가 2씩 증가하도록 (원하면 3,4 등으로 조절)
random.seed(seed)

def is_dir(p):
    return os.path.isdir(os.path.join(data_path, p))

def numeric_key(fn: str):
    # 파일명에 숫자가 있으면 그 숫자 기준으로 정렬, 없으면 문자열
    m = re.search(r'(\d+)', fn)
    return (int(m.group(1)) if m else fn, fn)

def max_take_with_stride(n: int, stride: int) -> int:
    """
    길이 n에서 stride 간격으로 '연속 구간'을 택할 때 최대 몇 개를 뽑을 수 있는지.
    예) n=10, stride=2 -> 인덱스 [s, s+2, s+4, s+6, s+8] 최대 5개
    """
    if n <= 0:
        return 0
    return (n - 1) // stride + 1

# 1) 디렉터리 목록
dirs = [d for d in os.listdir(data_path) if is_dir(d)]
dirs.sort()

# 2) 각 디렉터리의 json 목록(정렬)과 개수 수집
dir2files = {}
dir_sizes = []
for d in dirs:
    full = os.path.join(data_path, d)
    files = [f for f in os.listdir(full) if f.endswith('_heading.json')]
    files.sort(key=numeric_key)
    dir2files[d] = files
    dir_sizes.append(len(files))
# import pdb; pdb.set_trace()
# STRIDE를 고려한 '실제 선택 가능 용량'으로 분배
caps = [max_take_with_stride(n, STRIDE) for n in dir_sizes]
total_capacity = sum(caps)
if total_capacity < target_total:
    raise ValueError(
        f'요청 수({target_total})를 STRIDE={STRIDE}로는 충족할 수 없습니다. '
        f'전체 용량={total_capacity}. target_total을 줄이거나 STRIDE를 낮춰주세요.'
    )

# 3) 디렉터리별 기본 할당(비율), 나머지 분배(가장 큰 소수점부터) - 용량 caps 기준
fractions = []
alloc = []
total_caps = sum(caps)
for cap in caps:
    ratio = cap / total_caps * target_total
    base = floor(ratio)
    # cap(=해당 디렉토리에서 stride 간격으로 뽑을 수 있는 최대치)을 넘지 않게
    take0 = min(base, cap)
    alloc.append(take0)
    fractions.append(ratio - take0)

# 남은 개수 계산
left = target_total - sum(alloc)
if left > 0:
    # 소수점 큰 순으로 +1씩, 단 디렉터리의 남은 여유가 있어야 함
    candidates = [i for i in range(len(dirs)) if alloc[i] < caps[i]]
    candidates.sort(key=lambda i: fractions[i], reverse=True)
    j = 0
    while left > 0 and candidates:
        i = candidates[j % len(candidates)]
        if alloc[i] < caps[i]:
            alloc[i] += 1
            left -= 1
        j += 1
        if all(alloc[i] >= caps[i] for i in candidates):
            break

# 혹시 남았으면(이론상 드묾) 아직 여유 있는 디렉터리에 라운드로 분배
if left > 0:
    idxs = [i for i in range(len(dirs)) if alloc[i] < caps[i]]
    if not idxs:
        raise RuntimeError('분배할 남은 개수가 있으나 더 추가할 수 있는 디렉터리가 없습니다.')
    k = 0
    while left > 0:
        i = idxs[k % len(idxs)]
        if alloc[i] < caps[i]:
            alloc[i] += 1
            left -= 1
        k += 1

# 4) 각 디렉터리에서 '연속 구간 + STRIDE 간격'으로 랜덤 추출(랩어라운드 없이)
waymo_infos = []
for d, take in zip(dirs, alloc):
    if take == 0:
        continue
    files = dir2files[d]
    n = len(files)
    # 이 디렉터리에서 stride 간격으로 take개를 뽑을 수 있는지(안전 검사)
    cap = max_take_with_stride(n, STRIDE)
    if take > cap:
        # 이 경우는 위 분배 단계에서 cap을 넘지 않게 했으므로 이론상 발생 X
        take = cap

    if take <= 0:
        continue

    # 시작점 범위 계산:
    # start + (take-1)*STRIDE <= n-1  →  start <= n-1-(take-1)*STRIDE
    start_max = max(0, n - 1 - (take - 1) * STRIDE)
    start = random.randint(0, start_max) if start_max > 0 else 0

    # stride 간격으로 선택
    picked = [files[start + k * STRIDE] for k in range(take)]
    waymo_infos.extend([os.path.join(d, f) for f in picked])

assert len(waymo_infos) == target_total, f"샘플 결과 길이 {len(waymo_infos)} != {target_total}"

# 5) 파일로 저장
with open(text_path, 'w') as f:
    for p in waymo_infos:
        f.write(p + '\n')

print(f"샘플링 완료: {len(waymo_infos)}개 (STRIDE={STRIDE}) -> '{text_path}'")
