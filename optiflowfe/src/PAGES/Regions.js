
import NavBar from "../components/NavBar";
import { useEffect, useState } from "react";

import sggData from "../assets/data/sggdata.json";
import WaterOutFlowGraph from "../components/graph/WaterOutFlowGraph";
import DatePickerWithOption from "../components/datepicker/DatePickerWithOption";
export default function Regions() {
  const { kakao } = window;
  const [map, setMap] = useState(null);
  const [container, setContainer] = useState(null);
  const [graphTitle, setGraphTitle] = useState("J");
  const [dateOption, setDateOption] = useState(null);
  const [graphData, setGraphData] = useState(null);
  useEffect(() => {
    if (!dateOption) return;
    // console.log("🗺 [Regions] dateOption : ", dateOption);
    fetchWaterOutFlowData(dateOption);
  }, [dateOption, graphTitle]);


  // 배수지별 유출량 데이터
  const fetchWaterOutFlowData = async (dateOption) => {
    const url = `http://10.125.121.226:8080/api/reservoirdata/${dateOption.option}/${dateOption.selectedValue}/${graphTitle.toLowerCase()}`;
    console.log("🌊 [Regions] url : ", url);
    const resp = await fetch(url);
    const data = await resp.json();
    console.log("🌊 [Regions] 유출량 데이터 :", data);

    if (!data) return;

    const result = {};
    data.map((item, index) => {
      result[index + 1] = item.totalOutput.toFixed(2);
    });

    // console.log("🌊 [Regions] 유출량 데이터 graphData :", result);
    setGraphData(result);
  }

  // 지도 생성
  useEffect(() => {
    const containerOBJ = document.getElementById('map'); //지도를 담을 영역의 DOM 레퍼런스
    setContainer(containerOBJ);
    const options = { //지도를 생성할 때 필요한 기본 옵션
      center: new kakao.maps.LatLng(35.7992, 126.9260), //지도 첫 로드시 중심좌표.
      level: 10 //지도의 레벨(확대, 축소 정도)
    };

    const map = new kakao.maps.Map(containerOBJ, options); //지도 생성 및 객체 리턴
    setMap(map);

    // Polygon 데이터를 가져와 지도에 표시
    sggData.features.forEach((feature, index) => {
      if (feature.geometry.type === 'Polygon') {

        const path = feature.geometry.coordinates.map(ring =>
          ring.map(([lng, lat]) => new kakao.maps.LatLng(lat, lng))
        );

        // console.log("path :", path);
        const polygon = new window.kakao.maps.Polygon({
          map: map,
          path: path,
          strokeWeight: 3,
          strokeColor: "#777777",
          strokeOpacity: 0.8,
          fillColor: "#AAAAAA",
          fillOpacity: 0.6,
        });

        polygon.setMap(map);
      }
    });

    const markersInfo = [
      { type: "waterPlant", label: "정수장", positon: [35.9775480870027, 127.227240562789] },
      { type: "reservoir", label: "B", positon: [35.73, 126.92] },
      { type: "reservoir", label: "L", positon: [35.59, 126.81] },
      { type: "reservoir", label: "K", positon: [35.56, 126.92] },
      { type: "reservoir", label: "F", positon: [36.04, 127.01] },
      { type: "reservoir", label: "D", positon: [35.94, 126.96] },
      { type: "reservoir", label: "E", positon: [35.92, 126.75] },
      { type: "reservoir", label: "A", positon: [35.81, 126.82] },
      { type: "reservoir", label: "C", positon: [35.79, 127.03] },
      { type: "reservoir", label: "J", positon: [35.87, 127.31] },
      { type: "reservoir", label: "G", positon: [35.84, 127.13] },
      { type: "reservoir", label: "I", positon: [35.76, 127.19] },
      { type: "reservoir", label: "H", positon: [35.70, 127.16] },
    ];

    markersInfo.map((m) => {
      const markerimgSrc = m.type == "reservoir" ? "/images/marker_blue.png" : "/images/marker_red.png";
      const markerimageSize = m.type == "reservoir" ? new kakao.maps.Size(42, 45) : new kakao.maps.Size(50, 54); // 마커이미지의 크기
      const markerimageOption = { offset: new kakao.maps.Point(27, 69) }; // 마커이미지의 옵션입니다. 마커의 좌표와 일치시킬 이미지 안에서의 좌표를 설정합니다.

      const makerImage = new kakao.maps.MarkerImage(markerimgSrc, markerimageSize, markerimageOption);

      const positon = new kakao.maps.LatLng(m.positon[0], m.positon[1]);

      const marker = new kakao.maps.Marker({
        position: positon,
        image: makerImage
      });

      marker.setMap(map);


      // 글자
      const blueIwContent = `<div id=${m.label} class="pointer-events-none text-white relative bottom-10 right-[6px] font-bold" >${m.label}</div>`;
      const redIwContent = `<div class="px-4 py-1 text-xs relative top-3 left-[-3px] bg-white rounded-md ">${m.label}</div>`;

      const markerOverlay = new kakao.maps.CustomOverlay({
        map: map,
        position: positon,
        content: m.type == "reservoir" ? blueIwContent : redIwContent,
        yAnchor: 1
      });

      kakao.maps.event.addListener(marker, 'click', function () {
        // 마커 위에 인포윈도우를 표시합니다
        clickMarkers(m.label);
      });
    });


  }, []);

  const clickMarkers = (label) => {
    if (label == "정수장") return;
    console.log(label);
    setGraphTitle(label);
  };

  const Window = ({ label }) =>
    <div className="text-white font-bold"
      onClick={() => console.log("hello :", label)}>
      {label}
    </div>

  //주소를 지오코드로 변환
  const addrToGeo = () => {
    const geocoder = new kakao.maps.services.Geocoder();

    // 주소를 입력하세요
    const address = "전북 완주군 고산면 성재리 27";

    geocoder.addressSearch(address, (result, status) => {
      if (status === kakao.maps.services.Status.OK) {
        const coords = {
          lat: result[0].y,
          lng: result[0].x,
        };
        console.log("좌표:", coords);
      }
    }
    );
  }
  //지도 정보 얻어오는 함수
  const getMapInfo = () => {
    const center = map.getCenter();           // 지도의 현재 중심좌표를 얻어옵니다  
    const level = map.getLevel();             // 지도의 현재 레벨을 얻어옵니다
    const mapTypeId = map.getMapTypeId();     // 지도타입을 얻어옵니다
    const bounds = map.getBounds();           // 지도의 현재 영역을 얻어옵니다 
    const swLatLng = bounds.getSouthWest();   // 영역의 남서쪽 좌표를 얻어옵니다 
    const neLatLng = bounds.getNorthEast();   // 영역의 북동쪽 좌표를 얻어옵니다 

    let message = '지도 중심좌표는 위도 ' + center.getLat() + ', <br>';
    message += '경도 ' + center.getLng() + ' 이고 <br>';
    message += '지도 레벨은 ' + level + ' 입니다 <br> <br>';
    message += '지도 타입은 ' + mapTypeId + ' 이고 <br> ';
    message += '지도의 남서쪽 좌표는 ' + swLatLng.getLat() + ', ' + swLatLng.getLng() + ' 이고 <br>';
    message += '북동쪽 좌표는 ' + neLatLng.getLat() + ', ' + neLatLng.getLng() + ' 입니다';

    console.log(message);
  }

  return (
    <div className="w-fit min-[1530px]:w-full min-w-[1000px] h-screen bg-[#f2f2f2]">
      <NavBar />
      <div className="w-full h-screen pl-[260px] flex flex-col">
        <section className="w-full h-[160px] px-10 flex justify-between items-end">
          {/* 텍스트 */}
          <div className="w-2/5 h-full  flex flex-col justify-end text-[#333333]">
            <h1 className="text-4xl ">타이틀</h1>
            <p className="mt-2">각 배수지를 클릭하면, <span className="whitespace-nowrap"> 세부 정보를 확인할 수 있습니다. **멘트수정필요**</span></p>
          </div>
          <button onClick={getMapInfo} className="bg-blue-300 font-Freesentation font-light">지도정보보기</button>
          <DatePickerWithOption setDateOption={setDateOption} />
        </section>

        <div className="px-10 pb-10 pt-6 w-full h-full flex">
          {/* ===== 지도 section ===== */}
          <section className="w-full h-full min-w-[620px] pr-3 relative">
            <p className="absolute bottom-2 right-5 z-10 text-xs bg-white bg-opacity-80 p-1 rounded-sm">지도에 표시된 정수장 및 배수지의 위치는 시각적 참고를 위해 설정된 가상의 위치로, 실제 위치와 다를 수 있습니다.</p>
            <div id="map" className="w-full h-full border rounded-lg"></div>
          </section>

          <div className="pl-3 w-fit">
            {/* ===== 그래프1 ===== */}
            <section className="h-1/2 pb-4 w-[700px]">
              <div className="w-full h-full border-black bg-white flex justify-center items-center">
                {
                  graphData ?
                    <WaterOutFlowGraph graphTitle={graphTitle} data={graphData} datepickerOption={dateOption && dateOption.option} />
                    :
                    <div className="text-gray-500"> 날짜를 선택하세요 </div>
                }
              </div>
            </section>
            {/* ===== 그래프2 ===== */}
            <section className="h-1/2 pt-4 w-[700px]">
              <div className="w-full h-full border-black bg-white ">
              </div>
            </section>
          </div>
        </div>


      </div>
    </div>
  );
}
