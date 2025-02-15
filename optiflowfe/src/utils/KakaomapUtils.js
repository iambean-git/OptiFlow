//지도 정보 얻어오는 함수
const getMapInfo = (map) => {
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