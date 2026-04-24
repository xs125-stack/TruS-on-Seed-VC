import os
import re
import json
import time
import glob
import argparse
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import contextlib, io
import torch
from tqdm import tqdm

from inference_v2 import convert_voice_v2


SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def parse_speaker_and_utt(path: Path):
    """
    解析:
      ZH_B00000_S00010_W000000.mp3
    返回:
      speaker_id, utt_id
    """
    m = re.search(r"_S(\d+)_W(\d+)$", path.stem)
    if m is None:
        return None
    speaker_id = int(m.group(1))
    utt_id = int(m.group(2))
    return speaker_id, utt_id


def select_first_utterance_per_speaker(zh_dir: Path, n_speakers: int):
    """
    从 ZH_part 中:
    - 按 speaker_id 升序
    - 每个 speaker 取 utt_id 最小的一条（通常 W000000）
    """
    by_speaker = defaultdict(list)

    for p in zh_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        parsed = parse_speaker_and_utt(p)
        if parsed is None:
            continue
        speaker_id, utt_id = parsed
        by_speaker[speaker_id].append((utt_id, p))

    speaker_ids = sorted(by_speaker.keys())
    if len(speaker_ids) < n_speakers:
        raise ValueError(
            f"只找到 {len(speaker_ids)} 个可解析 speaker，少于要求的 {n_speakers} 个。"
        )

    selected = []
    for spk in speaker_ids[:n_speakers]:
        utts = sorted(by_speaker[spk], key=lambda x: x[0])
        selected.append((spk, utts[0][1]))

    return selected


def cache_to_summary(cache: dict):
    
    summary = {}
    eps = 1e-8
    for key, h in cache.items():
        if not torch.is_tensor(h):
            continue
        h = h.float()
        h_unit = h / (h.norm(p=2, dim=-1, keepdim=True) + eps)
        vec = h_unit.mean(dim=(0, 1))  # [H]
        summary[key] = vec.cpu()
    return summary


def build_prototype_from_summary_dir(summary_dir: Path):
    """
    从已有所有 summary 构建 prototype
   
    """
    summary_files = sorted(summary_dir.glob("*_summary.pt"))
    if len(summary_files) == 0:
        return {}, 0

    acc = {}
    counts = {}

    for path in summary_files:
        summary = torch.load(path, map_location="cpu")
        for key, vec in summary.items():
            if key not in acc:
                acc[key] = vec.clone()
                counts[key] = 1
            else:
                acc[key] += vec
                counts[key] += 1

    prototype = {k: acc[k] / counts[k] for k in acc}
    return prototype, len(summary_files)


def load_progress(progress_path: Path):
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "items": {},
        "meta": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }


def save_progress(progress_path: Path, progress: dict):
    tmp_path = progress_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    tmp_path.replace(progress_path)


def status_key_for_item(spk: int, path: Path):
    return f"S{spk:05d}|{path.stem}"


def make_infer_args(
    source_path: str,
    target_path: str,
    cache_out: str,
    output_dir: str,
    diffusion_steps: int,
    ar_checkpoint_path: str | None,
    cfm_checkpoint_path: str | None,
):
    """
    给 convert_voice_v2 构造参数对象
    """
    return SimpleNamespace(
        source=source_path,
        target=target_path,
        output=output_dir,
        diffusion_steps=diffusion_steps,
        length_adjust=1.0,
        compile=False,
        intelligibility_cfg_rate=0.7,
        similarity_cfg_rate=0.7,
        top_p=0.9,
        temperature=1.0,
        repetition_penalty=1.0,
        convert_style=False,
        anonymization_only=False,
        ar_checkpoint_path=ar_checkpoint_path,
        cfm_checkpoint_path=cfm_checkpoint_path,
        trus_enable=True,
        trus_mode="record",
        trus_profile=None,
        trus_alpha=0.8,  
        trus_alpha_branch0=0.4,
        trus_alpha_branch1=0.0,
        trus_alpha_branch2=0.0,
        trus_cache_out=cache_out,
    )


def inspect_first_summary(summary_path: Path):
    s = torch.load(summary_path, map_location="cpu")
    k = next(iter(s))
    print(f"[inspect] first summary key: {k}")
    print(f"[inspect] key len: {len(k)}")
    print(f"[inspect] num entries: {len(s)}")
    print(f"[inspect] shape: {s[k].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zh-dir", type=str, default="ZH_part")
    parser.add_argument("--probe-source", type=str, required=True)
    parser.add_argument("--n-speakers", type=int, default=50,
                        help="先用 50，后面改成 900 即可")
    parser.add_argument("--output-dir", type=str, default="./retain_3branch_out")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--rebuild-prototype-only", action="store_true")
    parser.add_argument("--ar-checkpoint-path", type=str, default=None)
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None)
    args = parser.parse_args()

    zh_dir = Path(args.zh_dir)
    probe_source = Path(args.probe_source)
    output_dir = Path(args.output_dir)

    if not zh_dir.exists():
        raise FileNotFoundError(f"找不到目录: {zh_dir}")
    if not probe_source.exists():
        raise FileNotFoundError(f"找不到 probe source: {probe_source}")

    cache_dir = output_dir / "caches"
    summary_dir = output_dir / "summaries"
    tmp_audio_dir = output_dir / "tmp_audio_out"
    progress_path = output_dir / "progress.json"
    prototype_out = output_dir / "prototype.pt"
    list_path = output_dir / "selected_retain_files.txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)

    selected = select_first_utterance_per_speaker(zh_dir, args.n_speakers)

    with open(list_path, "w", encoding="utf-8") as f:
        for spk, p in selected:
            f.write(f"{spk}\t{p}\n")

    progress = load_progress(progress_path)

    if args.rebuild_prototype_only:
        prototype, n_used = build_prototype_from_summary_dir(summary_dir)
        if n_used == 0:
            raise RuntimeError("没有 summary，无法重建 prototype。")
        torch.save(
            {
                "prototype": prototype,
                "n_speakers": n_used,
                "selected_files": [str(p) for _, p in selected],
                "probe_source": str(probe_source),
                "rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            prototype_out,
        )
        print(f"已重建 prototype: {prototype_out}")
        print(f"使用 summary 数量: {n_used}")
        first_key = next(iter(prototype))
        print(f"first key: {first_key}, shape: {prototype[first_key].shape}")
        return

    to_process = []
    done_count = 0
    failed_count = 0

    for spk, target_path in selected:
        item_key = status_key_for_item(spk, target_path)
        item_status = progress["items"].get(item_key, {}).get("status")

        if item_status == "done":
            done_count += 1
            continue
        if item_status == "failed" and not args.retry_failed:
            failed_count += 1
            continue
        to_process.append((spk, target_path))

    print(f"总 speaker 数: {len(selected)}")
    print(f"已完成: {done_count}")
    print(f"失败且本次跳过: {failed_count}")
    print(f"本次待处理: {len(to_process)}")
    print(f"清单已保存到: {list_path}")
    print(f"进度文件: {progress_path}")

    pbar = tqdm(
    to_process,
    desc="Building 3-branch prototype",
    unit="spk",
    dynamic_ncols=True,
    smoothing=0.1
    )

    first_done_summary = None

    for spk, target_path in pbar:
        item_key = status_key_for_item(spk, target_path)
        base_name = target_path.stem
        cache_out = cache_dir / f"{base_name}_cache.pt"
        summary_out = summary_dir / f"{base_name}_summary.pt"

        pbar.set_postfix({
    "spk": f"S{spk:05d}",
    "done": done_count
     })

        start = time.time()
        try:
            infer_args = make_infer_args(
                source_path=str(probe_source),
                target_path=str(target_path),
                cache_out=str(cache_out),
                output_dir=str(tmp_audio_dir),
                diffusion_steps=args.diffusion_steps,
                ar_checkpoint_path=args.ar_checkpoint_path,
                cfm_checkpoint_path=args.cfm_checkpoint_path,
            )

            # 这一步会调用 inference_v2 的 record 模式，生成三路 cache
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _ = convert_voice_v2(str(probe_source), str(target_path), infer_args)

            if not cache_out.exists():
                raise FileNotFoundError(f"没有生成 cache 文件: {cache_out}")

            cache = torch.load(cache_out, map_location="cpu")
            summary = cache_to_summary(cache)

            if len(summary) == 0:
                raise RuntimeError("summary 为空")

            torch.save(summary, summary_out)

            if first_done_summary is None:
                first_done_summary = summary_out

            if not args.keep_cache:
                try:
                    cache_out.unlink()
                except OSError:
                    pass

            elapsed = time.time() - start
            progress["items"][item_key] = {
                "status": "done",
                "speaker_id": spk,
                "target_path": str(target_path),
                "summary_path": str(summary_out),
                "elapsed_sec": elapsed,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_progress(progress_path, progress)

        except Exception as e:
            elapsed = time.time() - start
            progress["items"][item_key] = {
                "status": "failed",
                "speaker_id": spk,
                "target_path": str(target_path),
                "error": repr(e),
                "elapsed_sec": elapsed,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_progress(progress_path, progress)
            print(f"\n[FAILED] speaker={spk} file={target_path.name}")
            print(f"error: {repr(e)}")
            continue

    # 统一从 summary 目录重建三路 prototype
    prototype, n_used = build_prototype_from_summary_dir(summary_dir)
    if n_used == 0:
        raise RuntimeError("没有可用 summary，无法构建 prototype。")

    torch.save(
        {
            "prototype": prototype,
            "n_speakers": n_used,
            "selected_files": [str(p) for _, p in selected],
            "probe_source": str(probe_source),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        prototype_out,
    )

    done_total = sum(1 for v in progress["items"].values() if v.get("status") == "done")
    failed_total = sum(1 for v in progress["items"].values() if v.get("status") == "failed")

    print("\n全部流程结束。")
    print(f"prototype 已保存到: {prototype_out}")
    print(f"成功 summary 数量: {n_used}")
    print(f"progress done: {done_total}, failed: {failed_total}")

    first_key = next(iter(prototype))
    print(f"prototype entries: {len(prototype)}")
    print(f"first key: {first_key}, shape: {prototype[first_key].shape}")

    if first_done_summary is not None:
        inspect_first_summary(first_done_summary)


if __name__ == "__main__":
    main()