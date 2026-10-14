#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
waymo_e2e_training에서, 목록(txt)에 있는 항목만 골라서 새 폴더로 복사합니다.

목록(txt) 형식 예:
051b4464459ec44e35fe60785ee16a85/135_annotation.json
051ce5c30a4149e2e888f391cd01a211/087_annotation.json
...

기본 동작: annotation(.json)만 복사
옵션 --with-images: 같은 인덱스의 카메라 이미지도 함께 복사
  - 카메라 폴더 자동 탐색(하위 디렉토리 중 이미지가 있는 폴더)
  - 파일명 매칭은 0패딩 유무와 확장자(jpg/jpeg/png) 모두 고려
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def parse_list_line(line: str) -> Optional[Tuple[str, str, int]]:
    """
    'seq/NNN_annotation.json' -> (seq, 'NNN',  int(NNN))
    공백/주석 무시
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    # 허용 패턴: <seq>/<frame>_annotation.json
    m = re.match(r"^([^/]+)/(\d+)_annotation_heading\.json$", s)
    if not m:
        # 포맷이 다르면 스킵(필요시 raise로 바꿔도 됨)
        return None
    seq = m.group(1)
    frame_str = m.group(2)  # '085', '0030' 등
    frame_idx = int(frame_str)
    return seq, frame_str, frame_idx

def find_image_for_index(cam_dir: Path, frame_idx: int, frame_str: str) -> Optional[Path]:
    """
    주어진 인덱스에 해당하는 이미지 파일을 cam_dir에서 탐색.
    - 0패딩 유무 상관없이 매칭: ^0*{idx}\.(jpg|jpeg|png|bmp)$
    - 먼저 frame_str 그대로 시도(예: '085.jpg', '0030.jpg'), 그다음 정규식 검색
    """
    # 1) frame_str 그대로(원본 패딩 유지)
    for ext in IMG_EXTS:
        cand = cam_dir / f"{frame_str}{ext}"
        if cand.exists():
            return cand

    # 2) 정규식: 0*{idx}.ext
    idx = frame_idx
    pat = re.compile(rf"^0*{idx}\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
    try:
        for p in cam_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                if pat.match(p.name):
                    return p
    except FileNotFoundError:
        return None
    return None

def discover_camera_dirs(seq_dir: Path) -> List[Path]:
    """
    시퀀스 폴더 하위에서 이미지가 들어있는 하위 디렉터리(카메라 폴더)를 자동 탐색.
    """
    cams = []
    if not seq_dir.exists():
        return cams
    for child in seq_dir.iterdir():
        if child.is_dir():
            # 이미지가 하나라도 있으면 카메라 폴더로 간주
            has_img = any((child / f).is_file() and Path(f).suffix.lower() in IMG_EXTS
                          for f in os.listdir(child) or [])
            if has_img:
                cams.append(child)
    return cams

def copy_file(src: Path, dst: Path, dry_run: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY] copy {src} -> {dst}")
    else:
        shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser(description="Copy selected Waymo E2E items listed in a txt file.")
    ap.add_argument("--root", type=Path, required=True, help="원본 데이터 루트 (예: ./waymo_e2e_training)")
    ap.add_argument("--list", type=Path, required=True, help="선택 목록 txt 파일 경로")
    ap.add_argument("--out", type=Path, required=True, help="복사 대상 루트 (결과를 이곳에 생성)")
    ap.add_argument("--with-images", action="store_true", help="같은 인덱스의 카메라 이미지도 함께 복사")
    ap.add_argument("--cams", type=str, nargs="*", default=None,
                    help="카메라 폴더 이름 명시(예: front front_left front_right). 미지정 시 자동탐색")
    ap.add_argument("--dry-run", action="store_true", help="실제 복사 없이 무엇을 할지 출력만")
    args = ap.parse_args()

    root: Path = args.root
    out_root: Path = args.out
    list_path: Path = args.list

    copied_ann, missing_ann = 0, 0
    copied_img, missing_img = 0, 0

    if not list_path.exists():
        raise FileNotFoundError(f"목록 파일이 존재하지 않습니다: {list_path}")

    seen = set()  # 중복 라인 방지

    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_list_line(line)
            if parsed is None:
                continue
            seq, frame_str, frame_idx = parsed

            key = (seq, frame_str)
            if key in seen:
                continue
            seen.add(key)

            seq_dir = root / seq
            ann_src = seq_dir / f"{frame_str}_annotation_heading.json"
            ann_dst = out_root / seq / f"{frame_str}_annotation_heading.json"

            if ann_src.exists():
                copy_file(ann_src, ann_dst, args.dry_run)
                copied_ann += 1
            else:
                print(f"[WARN] annotation not found: {ann_src}")
                missing_ann += 1

            if args.with_images:
                # 카메라 폴더 획득
                if args.cams:
                    cam_dirs = [seq_dir / c for c in args.cams]
                else:
                    cam_dirs = discover_camera_dirs(seq_dir)

                if not cam_dirs:
                    print(f"[WARN] no camera folders under: {seq_dir}")
                for cam_dir in cam_dirs:
                    img_src = find_image_for_index(cam_dir, frame_idx, frame_str)
                    if img_src is None:
                        print(f"[WARN] image not found for index {frame_str} in {cam_dir}")
                        missing_img += 1
                        continue
                    rel_cam = cam_dir.name
                    img_dst = out_root / seq / rel_cam / img_src.name
                    copy_file(img_src, img_dst, args.dry_run)
                    copied_img += 1

    print("\n=== Summary ===")
    print(f"Annotations  - copied: {copied_ann}, missing: {missing_ann}")
    if args.with_images:
        print(f"Images       - copied: {copied_img}, missing: {missing_img}")
    print(f"Output root  : {out_root.resolve()}")
    if args.dry_run:
        print("[DRY RUN] 실제 복사는 수행하지 않았습니다.")

if __name__ == "__main__":
    main()
