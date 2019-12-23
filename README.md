# MyManagedCudaTest


## 개요
ManagedCuda 라이브러리 테스트 프로젝트 저장용

ManagedCuda 라이브러리는 C#에서 CUDA를 사용하는데 도움을 주는 라이브러리이다.

해당 라이브러리의 동작성과 사용방법을 시험해보기 위해 해당 프로젝트를 만들었다.

다만, 원작 프로젝트인 [ManagedCuda](http://kunzmi.github.io/managedCuda/)에서 Fork된 프로젝트인 [UnmanageableCuda](https://github.com/kaby76/managedCuda)를 이용했는데, 큰 차이는 없을 것이라 생각한다.



### 사용한 개발 환경
- Visual studio 2017
- CUDA 10.0 ← 만일 문제가 있으면 9.2로 재시도 해볼 것



### 사용한 Nuget 패키지
- Microsoft.NETCore.Platforms.3.1.0
- NETStandard.Library.2.0.3
- UnmanageableCuda.10.130.1



### 프로젝트 기본 구조
- MyCudaPtxRunner
  - BuildOutPut
  - MyCudaPtxRunner
  - MyPtx ← VS2015
  - MyPtx2 ← VS2015
  - packages
  - MyCudaPtxRunner.sln



#### PTX파일 만들기
1. CUDA프로젝트를 만들어서 Build output을 ptx 설정
  - [참고링크 1](https://blog.naver.com/pkk1113/221362455788)
  - [참고링크 2](https://blog.naver.com/pjm2108/220269762506)
2. 함수를 이용하여 C또는 CU파일을 읽음
  - [SimpleCUDAExample](https://github.com/mgravell/SimpleCUDAExample)
  - 자신의 CUDA 버전에 맞는 dll이 필요하다.
    - 예1) nvrtc64_100_0.dll ← 단, CUDA 10.0일 경우 이름을 nvrtc64_100.dll로 변경할 것
    - 예2) nvrtc-builtins64_100.dll



### 이슈
#### Visual studio와 CUDA간 호환성
간혹 CUDA가 Visual studio 2017에서 제대로 실행되지 않는 문제가 있다.
host_config.h 파일을 수정하거나 프로젝트 플랫폼 도구에서 사용하는 VS 버전을 다운그레이드하면 된다.
현재 이 프로젝트에 있는 MyPtx와 MyPtx2는 후자의 방법을 사용했다.

####
