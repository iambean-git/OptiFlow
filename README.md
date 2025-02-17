# 💧 OPTIFLOW
[K-Digital 부산대 8회차] AI 활용 빅데이터분석 풀스택웹서비스 SW 개발자 양성과정 AI학습모델 웹서비스 개발 프로젝트
- **주제:** 상수도 시설 모니터링 및 배수지 수요량 예측을 통한 운영 최적화 

![OPTIFLOW](./assets/thumbnail.jpg)

<br/>

## 📈 개발 기간
> 2024.01.13 - 2025.02.20.

<br/>

## 👥 팀원 구성
<!-- |이름|담당|GITHUB|
|:------:|:---:|---|
|조은빈|FRONT-END|[![이미지](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/iambean-git)|
|정원영|BACK-END|[![이미지](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/wonny725)|
|윤찬희|DATA|[![이미지](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/chanheeYun)| -->

|조은빈|정원영|윤찬희|
|:---:|:---:|:---:|
|<img src="./assets/face/face_eb.png" width="130" height="130" alt="은빈" />|<img src="./assets/face/face_wy.png" width="130" height="130" alt="원영" />|<img src="./assets/face/face_ch.png" width="130" height="130" alt="찬희" />|
|FRONT-END|BACK-END|DATA|
|[![이미지](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/iambean-git)|[![이미지](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/wonny725)| [![이미지](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/chanheeYun)|

<br/>

## 🔧 Stack

### **FRONT-END**
![html](https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white) &nbsp; ![css](https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white) &nbsp; ![js](https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black) &nbsp; ![react](https://img.shields.io/badge/react-61DAFB?style=for-the-badge&logo=react&logoColor=black) &nbsp; ![tailwindcss](https://img.shields.io/badge/tailwindcss-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=black)

### **BACK-END**
![java](https://img.shields.io/badge/java-007396?style=for-the-badge&logo=java&logoColor=white) &nbsp; ![springboot](https://img.shields.io/badge/springboot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white) &nbsp; ![mysql](https://img.shields.io/badge/mysql-4479A1?style=for-the-badge&logo=mysql&logoColor=white) &nbsp; ![swagger](https://img.shields.io/badge/swagger-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)

### **DATA**
![python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white) &nbsp; ![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) &nbsp; ![numpy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white) &nbsp; ![scikitlearn](https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) &nbsp; ![pytorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) &nbsp; ![fastapi](https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white)

### **COMMON**
![git](https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white) &nbsp; ![github](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white) &nbsp; ![notion](https://img.shields.io/badge/notion-00148C?style=for-the-badge&logo=notion&logoColor=white) &nbsp; ![figma](https://img.shields.io/badge/figma-F24E1E?style=for-the-badge&logo=figma&logoColor=white)

<br/>

## ⭐ Main Feature
### 1️⃣ 메인 화면
<img src="./assets/capture/main1.gif" width="45%"  alt="main1" /> &nbsp; &nbsp; <img src="./assets/capture/main2.GIF" width="45%"  alt="main2" />
<img src="./assets/capture/main3.gif" width="45%"  alt="main3" /> &nbsp; &nbsp; <img src="./assets/capture/main4.GIF" width="45%"  alt="main4" />

- Optiflow 주요 서비스 소개

<br>

### 2️⃣ 이용 신청 / 로그인 / 비밀번호 변경
<img src="./assets/capture/inquiry.GIF" width="31%"  alt="inquiry" /> &nbsp; <img src="./assets/capture/capture_approved_mail.png" width="31%"  alt="main2" /> &nbsp; <img src="./assets/capture/capture_pwChange.png" width="31%"  alt="inquiry" />

- 메인 화면 하단에서 이용 문의 신청 가능 (모달로 구현)
- 이용 신청 후, 관리자 승인 시 이메일을 통해 임시비밀번호 발급 및 안내
- 로그인 후 '내 계정 관리' 탭에서 비밀번호 변경 가능

<br>

### 3️⃣ 대시보드
![대시보드](./assets/capture/capture_dashboard.png)
- 공통 : 아이콘 클릭 및 SelectBox를 통한 배수지 변경
- 영역 1 : 전체 배수지 현재 수위 실시간 안내
- 영역 2 : 선택된 배수지의 상세 정보 제공 (수위, 현재 저수량, 1시간 뒤 예상 저수량)
- 영역 3 : 지난 24시간 유입량 및 유출량 확인
- 영역 4 : 추후 24시간 예측 유출량 및 추천 유입량 안내 (XG Boost / LSTM 모델 변경 가능)

<br>

### 4️⃣ 지난 수위 정보 조회
![waterFlow](./assets/capture/mulmung.gif)
- 모식도 형태로 한 눈에 볼 수 있는 흐름도 제공 (React-Flow 활용)
- 과거 날짜 및 시간 선택시, 해당일의 수위 데이터 안내 
- 각 배수지 hover시, 상세 데이터 안내

<br>

### 5️⃣ 배수지별 통계 정보
![regions](./assets/capture/capture_regions.png)
- Kakao Map API를 활용한 위치 안내 
    - 시각적 참고로 위해 가상으로 설치된 위치로, 실제 위치와 다를 수 있습니다.
- 마커 클릭시, 선택되는 배수지 변경
- 시간별, 일별, 월별 통계 정보를 볼 수 있음
- 그래프 1 : 실제 유출량 및 AI모델로 예측된 유출량
- 그래프 2 : 실측 기반 전기 사용량과 AI모델 사용시 예측 되는 전기 사용량

<br>

### 6️⃣ 관리자 페이지
- 

## 🔨 Server Architecture

## 📋 ERD

## 📋 느낀점