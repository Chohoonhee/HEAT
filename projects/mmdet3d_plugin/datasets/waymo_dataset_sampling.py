from typing import Sequence, List, Tuple, Optional, Union
import random

def sample_even_count_near_interval(
    data: Sequence,
    target_n: int,
    approx_step: Optional[int] = None,
    include_last: bool = True,
    pattern: Union[str, tuple, List[int]] = "even",
) -> Tuple[List[int], List]:
    """
    data에서 거의 일정한 간격으로 정확히 target_n개 샘플.
    간격은 base 또는 base+1이며, +1을 줄 위치 분배를 pattern으로 '결정적'으로 제어.

    pattern:
      - "even"       : +1 간격을 균등 간격으로 흩뿌림(기본, 결정적)
      - "front"      : 앞쪽부터 num_big개에 +1
      - "back"       : 뒤쪽부터 num_big개에 +1
      - "alternate"  : 거의 교대로 +1
      - ("hash", s)  : seed=s 로 의사난수지만 항상 같은 결과
      - [list[int]]  : 길이 num_big인, 0..gaps-1 범위의 고유 인덱스 직접 지정
    """
    M = len(data)
    if target_n <= 0:
        return [], []
    if target_n == 1:
        idx = M - 1 if include_last else 0
        return [idx], [data[idx]]
    if target_n > M:
        raise ValueError(f"target_n({target_n}) > data length({M})")

    gaps = target_n - 1
    span = (M - 1) if include_last else max(M - 1 - (M // (gaps + 1)), 1)

    avg = span / gaps
    base = int(avg)
    num_big = int(round((avg - base) * gaps))
    num_big = max(0, min(gaps, num_big))

    # 시작점(미세조정은 결정적)
    start = 0
    if approx_step is not None and approx_step > 0:
        candidate_starts = range(0, max(1, base + 1))
        best = (float("inf"), 0)
        for s in candidate_starts:
            total_jump = base * gaps + num_big
            last_idx = s + total_jump
            if include_last:
                if last_idx > M - 1:
                    continue
            else:
                if last_idx >= M:
                    continue
            diff = abs(avg - approx_step)
            if diff < best[0]:
                best = (diff, s)
        start = best[1]

    # --- +1 간격을 줄 위치들을 '결정적'으로 선택 ---
    big_positions = _choose_big_gap_positions(gaps, num_big, pattern)

    # 간격 나열 생성 (길이 = gaps)
    gap_list = []
    for i in range(gaps):
        gap_list.append(base + (1 if i in big_positions else 0))

    # 인덱스 누적
    indices = [start]
    for g in gap_list:
        indices.append(indices[-1] + g)

    # include_last면 끝을 정확히 M-1로 정렬
    if include_last:
        shift = (M - 1) - indices[-1]
        if shift != 0:
            indices = [i + shift for i in indices]

    # 경계/단조 보정
    indices = [max(0, min(M - 1, i)) for i in indices]
    last = -1
    for i, v in enumerate(indices):
        if v <= last:
            v = last + 1
            if v > M - 1:
                v = M - 1
        indices[i] = v
        last = v

    # 드물게 길이 보정 필요 시 정수 linspace fallback
    if len(indices) != target_n:
        indices = [round(i) for i in __int_linspace(0, M - 1 if include_last else M - 1 - base, target_n)]

    samples = [data[i] for i in indices]
    return indices, samples


def _choose_big_gap_positions(
    gaps: int, num_big: int, pattern: Union[str, tuple, List[int]]
) -> List[int]:
    if num_big == 0:
        return []
    if isinstance(pattern, list):
        # 사용자가 직접 지정
        pos = sorted(set(pattern))
        if len(pos) != num_big or (pos and (pos[0] < 0 or pos[-1] >= gaps)):
            raise ValueError("big_gap_positions 불일치: 길이/범위 확인")
        return pos

    if isinstance(pattern, tuple):
        name, *rest = pattern
        name = str(name).lower()
        if name == "hash":
            seed = rest[0] if rest else 0
            rng = random.Random(int(seed))
            # 결정적으로 고르게 섞인 위치 선택(중복 없이)
            # 균등 후보를 만들어 놓고 deterministically shuffle
            pos = _even_spread_positions(gaps, num_big)
            rng.shuffle(pos)
            return sorted(pos)
        else:
            raise ValueError(f"알 수 없는 pattern tuple: {pattern}")

    name = str(pattern).lower()
    if name == "even":
        return _even_spread_positions(gaps, num_big)
    if name == "front":
        return list(range(num_big))
    if name == "back":
        return list(range(gaps - num_big, gaps))
    if name == "alternate":
        # 거의 교대로: 균등 분산과 유사하되 0부터 박자 맞추기
        step = gaps / num_big
        return sorted({int(round(k * step)) for k in range(num_big)} - {gaps})  # 끝 범위 초과 방지

    raise ValueError(f"알 수 없는 pattern: {pattern}")


def _even_spread_positions(gaps: int, num_big: int) -> List[int]:
    """
    0..gaps-1 구간에서 num_big개의 위치를 균등하게 뽑는 '결정적' 규칙.
    중앙 오프셋을 써서 클러스터링 최소화.
    수식: floor((gaps * (2k+1)) / (2*num_big))   (k=0..num_big-1)
    """
    pos = []
    denom = 2 * num_big
    used = set()
    for k in range(num_big):
        p = (gaps * (2 * k + 1)) // denom
        if p >= gaps:
            p = gaps - 1
        # 혹시나 충돌 시 +1씩 전진(결정적)
        while p in used and p + 1 < gaps:
            p += 1
        used.add(p)
        pos.append(p)
    return sorted(pos)


def __int_linspace(start: int, stop: int, num: int) -> List[int]:
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [int(round(start + i * step)) for i in range(num)]
