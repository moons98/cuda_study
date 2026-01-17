# CUDA Kernel Benchmark Framework

CUDA 커널 성능을 분석하고 비교하는 벤치마크 프레임워크입니다. SGEMM(Single-precision General Matrix Multiply) 최적화 기법을 단계별로 구현하고 성능을 측정합니다.

## 주요 기능

- **커널 벤치마크**: 다양한 SGEMM 구현의 성능 측정
- **커널 비교**: 여러 커널을 동시에 비교하고 cuBLAS 대비 효율성 분석
- **Roofline 분석**: 메모리/연산 바운드 분석 및 시각화
- **프로파일링**: Nsight Compute/Systems 명령어 자동 생성

## 구현된 커널

| 커널 | 설명 |
|------|------|
| `naive` | 기본 구현 (Global Memory 직접 접근) |
| `coalesced` | Memory Coalescing 최적화 |
| `shared_mem` | Shared Memory 타일링 |
| `cublas` | cuBLAS 라이브러리 (baseline) |

## 요구사항

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 지원 컴파일러
- NVIDIA GPU (Compute Capability 7.5+)

## 빌드

### Windows
```bash
build.bat
```

### Linux/macOS
```bash
chmod +x build.sh
./build.sh
```

### 수동 빌드
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## 사용법

### 단일 커널 벤치마크
```bash
./build/bin/benchmark --single --kernel=naive --size=1024
```

### 커널 비교
```bash
./build/bin/benchmark --compare --sizes=512,1024,2048
```

### Roofline 분석
```bash
./build/bin/benchmark --roofline --sizes=512,1024,2048,4096
```

### 전체 분석
```bash
./build/bin/benchmark --all --output=./results
```

### 사용 가능한 커널 목록
```bash
./build/bin/benchmark --list
```

## 명령줄 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--single` | 단일 커널 벤치마크 모드 | (기본 모드) |
| `--compare` | 커널 비교 모드 | - |
| `--roofline` | Roofline 분석 모드 | - |
| `--profile` | 프로파일링 명령어 생성 | - |
| `--all` | 모든 분석 실행 | - |
| `--kernel=NAME` | 벤치마크할 커널 이름 | naive |
| `--kernels=A,B,C` | 비교할 커널 목록 | (전체) |
| `--size=N` | 행렬 크기 | - |
| `--sizes=A,B,C` | 행렬 크기 목록 | 128,256,512,1024,2048,4096 |
| `--baseline=NAME` | 비교 기준 커널 | cublas |
| `--warmup=N` | 워밍업 횟수 | 5 |
| `--runs=N` | 벤치마크 반복 횟수 | 50 |
| `--verify` | 결과 검증 활성화 | on |
| `--no-verify` | 결과 검증 비활성화 | - |
| `--output=PATH` | 출력 디렉토리 | ./output |
| `--format=FMT` | 출력 형식 (console, csv) | console |
| `--device=N` | CUDA 디바이스 ID | 0 |

## 프로젝트 구조

```
CUDA_study/
├── include/
│   ├── core/           # 코어 기능 (타이머, 디바이스 정보, 벤치마크)
│   ├── utils/          # 유틸리티 (행렬 연산, 검증, CSV 내보내기)
│   ├── analysis/       # 분석 도구 (비교, Roofline, 프로파일링)
│   └── kernels/        # 커널 레지스트리
├── src/
│   ├── core/
│   ├── utils/
│   ├── analysis/
│   ├── kernels/
│   └── main.cu         # 메인 진입점
├── kernels/
│   └── sgemm/          # SGEMM 커널 구현
├── output/             # 벤치마크 결과 출력
├── CMakeLists.txt
├── build.bat           # Windows 빌드 스크립트
└── build.sh            # Linux/macOS 빌드 스크립트
```

## 출력 예시

### 커널 비교 테이블
```
================================================================================
KERNEL COMPARISON RESULTS
================================================================================
Size     | naive      | coalesced  | shared_mem | cublas
---------|------------|------------|------------|------------
512      | 45.2 GFLOPS| 89.4 GFLOPS| 156.7 GFLOPS| 423.5 GFLOPS
1024     | 52.1 GFLOPS| 112.3 GFLOPS| 287.6 GFLOPS| 892.4 GFLOPS
2048     | 48.7 GFLOPS| 134.5 GFLOPS| 412.8 GFLOPS| 1245.6 GFLOPS
```

### Roofline 분석
- 메모리 대역폭 한계선
- 컴퓨팅 성능 한계선
- 각 커널의 Operational Intensity와 성능 위치

## 라이선스

MIT License
