import { RiMapPinFill } from "react-icons/ri";
import NavBar from "../components/NavBar";
import { useEffect, useState } from "react";

import sggData from "../assets/data/sggdata.json";

export default function Regions() {
  const { kakao } = window;
  const [map, setMap] = useState(null);
  const [container, setContainer] = useState(null);


  // 지도 생성
  useEffect(() => {
    const containerOBJ = document.getElementById('map'); //지도를 담을 영역의 DOM 레퍼런스
    setContainer(containerOBJ);
    const options = { //지도를 생성할 때 필요한 기본 옵션
      center: new kakao.maps.LatLng(35.8044830568065, 126.91), //지도의 중심좌표.
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

    const imgSrc = "/images/maker_red.png";
    const imageSize = new kakao.maps.Size(64, 69) // 마커이미지의 크기입니다
    const imageOption = { offset: new kakao.maps.Point(27, 69) }; // 마커이미지의 옵션입니다. 마커의 좌표와 일치시킬 이미지 안에서의 좌표를 설정합니다.

    const markerImage = new kakao.maps.MarkerImage(imgSrc, imageSize, imageOption);
    const markerPosition = new kakao.maps.LatLng(35.9775480870027, 127.227240562789); // 마커가 표시될 위치입니다

    //마커 생성
    const marker = new kakao.maps.Marker({
      position: markerPosition,
      image: markerImage // 마커이미지 설정 
    });

    // 마커가 지도 위에 표시되도록 설정합니다
    marker.setMap(map);


    
  }, []);


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
    // 지도의 현재 중심좌표를 얻어옵니다 
    const center = map.getCenter();

    // 지도의 현재 레벨을 얻어옵니다
    const level = map.getLevel();

    // 지도타입을 얻어옵니다
    const mapTypeId = map.getMapTypeId();

    // 지도의 현재 영역을 얻어옵니다 
    const bounds = map.getBounds();

    // 영역의 남서쪽 좌표를 얻어옵니다 
    const swLatLng = bounds.getSouthWest();

    // 영역의 북동쪽 좌표를 얻어옵니다 
    const neLatLng = bounds.getNorthEast();

    // 영역정보를 문자열로 얻어옵니다. ((남,서), (북,동)) 형식입니다
    const boundsStr = bounds.toString();


    let message = '지도 중심좌표는 위도 ' + center.getLat() + ', <br>';
    message += '경도 ' + center.getLng() + ' 이고 <br>';
    message += '지도 레벨은 ' + level + ' 입니다 <br> <br>';
    message += '지도 타입은 ' + mapTypeId + ' 이고 <br> ';
    message += '지도의 남서쪽 좌표는 ' + swLatLng.getLat() + ', ' + swLatLng.getLng() + ' 이고 <br>';
    message += '북동쪽 좌표는 ' + neLatLng.getLat() + ', ' + neLatLng.getLng() + ' 입니다';

    console.log(message);
  }


  return (
    <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2]">
      <NavBar />
      <div className="w-full h-screen pl-[260px] flex flex-col">
        <section className="w-full h-[160px] px-10 flex justify-between ">
          <div className="h-full flex items-end">
            <p className="text-xs mb-[-1rem]">상기 배수장 및 배수지의 위치 정보는 참고용으로 제공된 것으로, 실제 위치와는 차이가 있을 수 있습니다.</p>
          </div>
          <button onClick={getMapInfo} className="bg-blue-300">지도정보보기</button>
        </section>
        <section className="px-10 pb-10 pt-6 w-2/3 h-full">
          <div id="map" className="w-full h-full border rounded-lg"></div>
        </section>
      </div>
    </div>
  );
}
