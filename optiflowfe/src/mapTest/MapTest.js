import { Container as MapDiv, NaverMap, Marker } from "react-naver-maps";

function MapTest() {
  <MapDiv
    style={{
      height: 400,
    }}
  >
    왜 안되지?
    <NaverMap>
      <Marker defaultPosition={{ lat: 37.5666103, lng: 126.9783882 }} />
    </NaverMap>
  </MapDiv>;
}

export default MapTest;
