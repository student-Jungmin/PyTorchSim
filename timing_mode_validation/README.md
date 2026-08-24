# timing_mode_validation

TOGSim의 사이클을 실물 TPU 사이클과 맞추기 위한 마이크로벤치 모음.
케이스는 **연산별로** `ops/<연산>.py`에 있고, 그 파일이 유일한 정의처다.

```
ops/<op>.py      연산 하나의 케이스 목록 (name, size, origin, census, source, note)
bench.py         연산 32종의 실제 실행 -- 한 번에 하나, npu:0 + torch.compile + TOGSimulator
cases.py         ops/*.py를 읽어 한 목록으로 (필터: op, origin)
roofline.py      MAC/DRAM 바이트/ridge -- v3와 v6e를 configs/에서 읽은 값으로
run.py           케이스마다 프로세스 하나, 로그 하나
index.py         ops/*.py -> timing_cases.csv + PARAMS.md
PARAMS.md        연산별 파라미터 범위 (보고용, 생성물)
summary.py       로그 + 케이스 + 레퍼런스 조인
tpu_ref.py       TPU 쪽 JAX 측정기 (하드웨어가 없어 미검증)
logs/            <workload>.log, fp32가 아니면 <workload>.<dtype>.log
```

## 돌리기

기본 머신은 **v6e**다 (`systolic_ws_256x256_c1_simple_noc_tpuv6e_timing_only.yml`).
다른 기계로 재려면 `TOGSIM_CONFIG`를 잡고 돌린 뒤 `summary.py --machine v3`처럼 맞춰 준다.

```bash
source /workspace/tnpu-env.sh

python timing_mode_validation/run.py --op gemm --list
python timing_mode_validation/ops/gemm.py --dtype float32 --jobs 4      # 연산 하나만
python timing_mode_validation/run.py --dtype float16 --jobs 4           # 전부
python timing_mode_validation/run.py --op rmsnorm --origin added        # 추가한 구간만
python timing_mode_validation/summary.py --op rmsnorm --machine v6e
```

`run.py`가 `TORCHSIM_TIMING_MODE=1`을 스스로 켠다 — 이 브랜치는 타이밍이 기본 OFF라서,
없으면 사이클이 1로 나온다. `--isolate`는 케이스마다 dump path를 따로 준다(숫자가
이상할 때 커널 캐시를 배제하는 용도).

## 케이스를 고치거나 더할 때

`ops/<연산>.py`의 `CASES`에 한 줄 넣고 `python timing_mode_validation/index.py` —
`timing_cases.csv`와 `PARAMS.md`가 같이 다시 만들어진다.
`origin`은 그 shape의 출처다 — `census`(캡처된 모델의 층) / `sweep`(축 하나만 변형) /
`anchor`(기존 TPUv3 수치가 있는 것) / `added`(모델에 없지만 자주 쓰는 값).
`census`는 `/workspace/op-census/aggregate.json`에 대조한 결과이고 손으로 적지 않는다.

## 폭

fp32와 fp16 두 번 돌린다.

**fp16은 spike를 끈 상태에서만 돈다.** 이 빌드의 spike에는 `zvfh`가 없어서 벡터 fp16
명령을 실행하지 못한다 — 타이밍 전용 실행에는 상관없지만, `TOGSIM_CONFIG`가
functional mode를 켜는 config(예: `..._tpuv6e.yml`)를 가리키고 있으면 fp16이 spike에서
죽는다. `run.py`는 `_timing_only` config를 기본으로 잡지만 이미 설정된 `TOGSIM_CONFIG`를
덮지 않으므로, `tnpu-env.sh`를 source한 셸에서는 직접 지정해야 한다:

```bash
export TOGSIM_CONFIG=$TORCHSIM_DIR/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_timing_only.yml
```

**bf16은 백엔드가 거부한다** —
`tnpu/passes/p00_refuse_narrow_floats.py`가 "bf16 has no instruction on this machine"으로
멈춘다. TPU 쪽은 bf16이 네이티브이므로 2바이트 열은 **우리 fp16 대 TPU bf16**으로 짝짓고,
보고서에 그 치환을 적는다.
