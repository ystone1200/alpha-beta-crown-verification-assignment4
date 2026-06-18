# 과제 4: alpha-beta-CROWN을 이용한 Wine MLP 검증

이 저장소는 과제 3에서 Marabou로 검증했던 Wine dataset 기반 MLP를
외부 ONNX 모델로 가져와 alpha-beta-CROWN에서 검증하고, 두 도구의 결과를
비교한 과제 4 제출물이다.

## 구성 파일

- `report.pdf`: 최종 보고서.
- `docs/problem1_alpha_beta_crown_models.md`: alpha-beta-CROWN 공식 저장소의
  `models` 및 `exp_configs` 디렉터리 탐색 정리.
- `models/wine_mlp.onnx`: 과제 3에서 사용한 외부 ONNX 모델.
- `models/wine_mlp_metadata.json`: 모델 구조, scaler, Wine dataset 관련 메타데이터.
- `data/wine_verification_cases.json`: 검증에 사용할 Wine test sample과
  Marabou baseline 결과.
- `specs/wine/*.vnnlib`: sample/epsilon별 VNNLIB unsafe-region 명세.
- `configs/wine_mlp_base.yaml`: alpha-beta-CROWN 실행용 기본 YAML 설정 파일.
- `scripts/generate_wine_specs.py`: Wine sample과 epsilon 설정으로 VNNLIB 파일을
  생성하는 스크립트.
- `test.py`: alpha-beta-CROWN을 실행하고 Marabou 결과와 비교하는 실행 스크립트.
- `results/wine_abcrown_results.json`: alpha-beta-CROWN 실행 결과와 비교 결과.
- `requirements.txt`: 이 저장소의 보조 스크립트 실행에 필요한 Python 패키지.

## 환경 설정

alpha-beta-CROWN 자체는 로컬 도구 디렉터리로 설치해서 사용하며, 크기가 크고
외부 저장소이므로 이 저장소에는 커밋하지 않았다. 저장소 루트에서 다음과 같이
설치할 수 있다.

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
uv sync --python 3.11
cd ..
```

실험에는 alpha-beta-CROWN commit
`746b7d0128df1806c92381d1c8b3a66c9cba990c`를 사용하였다. Windows 환경에서는
`auto_LiRPA` submodule import 경로 문제가 있어, `test.py`에서
`alpha-beta-CROWN/auto_LiRPA`를 `PYTHONPATH` 앞에 추가한 뒤 verifier를 실행한다.

보조 패키지는 다음과 같이 설치한다.

```bash
pip install -r requirements.txt
```

## 실행 방법

필요하면 VNNLIB 명세 파일을 다시 생성한다.

```bash
python scripts/generate_wine_specs.py
```

기본 검증 sample인 `low_margin_3`에 대해 alpha-beta-CROWN을 실행한다.

```bash
python test.py
```

기본 실행은 정규화된 Wine feature 공간에서 epsilon
`0.01, 0.05, 0.1, 0.3, 0.5, 1.0`을 sweep하고, 결과를
`results/wine_abcrown_results.json`에 저장한다.

저장된 모든 sample을 실행하려면 다음 옵션을 사용한다.

```bash
python test.py --all-samples
```

특정 epsilon만 실행할 수도 있다.

```bash
python test.py --epsilons 0.01 0.1 0.3
```

## 검증 속성

검증 대상은 `sklearn.datasets.load_wine()`의 13개 feature를
`StandardScaler`로 정규화한 입력을 받는 MLP이다. 모델 구조는 다음과 같다.

```text
13 -> 16 -> ReLU -> 8 -> ReLU -> 3
```

선택한 test input `x`와 반경 `epsilon`에 대해 VNNLIB 파일은 다음 unsafe
condition을 표현한다.

```text
exists x' such that |x'_i - x_i| <= epsilon
and some non-original class logit >= the original predicted class logit
```

따라서 alpha-beta-CROWN 결과는 다음과 같이 해석하였다.

- `unsat`: unsafe condition에 도달할 수 없으므로 해당 epsilon에서 verified.
- `sat`: 반례가 존재하므로 해당 epsilon에서 falsified.
- `timeout`: 제한 시간 안에 결론에 도달하지 못함.

## 실험 결과 요약

기본 sample `low_margin_3`에서는 alpha-beta-CROWN과 Marabou의 판단이 모든
epsilon에서 일치하였다.

| epsilon | alpha-beta-CROWN | Marabou | method |
| ---: | --- | --- | --- |
| 0.01 | verified | verified | CROWN |
| 0.05 | verified | verified | CROWN |
| 0.10 | verified | verified | CROWN |
| 0.30 | falsified | falsified | PGD |
| 0.50 | falsified | falsified | PGD |
| 1.00 | falsified | falsified | PGD |

실험한 epsilon sweep 기준으로 이 sample은 `epsilon <= 0.10`에서는 robust하고,
`epsilon >= 0.30`에서는 반례가 발견되었다. 자세한 실행 시간과 로그 요약은
`results/wine_abcrown_results.json`과 `report.pdf`에 정리되어 있다.
