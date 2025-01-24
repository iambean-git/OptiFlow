import {
  Container as MapDiv,
  NaverMap,
  Marker,
  useNavermaps,
  Polygon,
  Polyline,
} from "react-naver-maps";

function MapTest() {
  const navermaps = useNavermaps();
  // 35.783717, 127.139220 지곡 배수지
  const polygonPath = [
    new navermaps.LatLng(35.784172, 127.139167),
    new navermaps.LatLng(35.784172, 127.140613),
    new navermaps.LatLng(35.78253, 127.140648),
    new navermaps.LatLng(35.78253, 127.139079),
  ];

  const polylinePath = [
    new navermaps.LatLng(35.784172, 127.139167),
    new navermaps.LatLng(35.784172, 127.140613),
    new navermaps.LatLng(35.78253, 127.140648),
    new navermaps.LatLng(35.78253, 127.139079),
  ];

  return (
    <MapDiv
      style={{
        height: "800px",
      }}
    >
      <NaverMap
        defaultCenter={new navermaps.LatLng(35.982984, 127.217228)}
        defaultZoom={14}
      >
        <Polygon
          paths={polygonPath}
          fillColor={"#ff0000"} // 채우기 색상
          fillOpacity={0.5} // 채우기 불투명도
          strokeColor={"#ff0000"} // 테두리 색상
          strokeOpacity={0.8} // 테두리 불투명도
          strokeWeight={3} // 테두리 두께
        />
        <Polyline
          paths={polylinePath}
          strokeColor={"#FF0000"} // 선 색상
          strokeOpacity={0.8} // 선 불투명도
          strokeWeight={5} // 선 두께
        />
      </NaverMap>
    </MapDiv>
  );
}

export default MapTest;
