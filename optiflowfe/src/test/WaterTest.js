// import { Liquid } from '@ant-design/plots';

// export default function WaterTest({percent, size}) {
//     const config = {
//         percent: percent,
//         liquidStyle: {
//             fill: '#4a90e2', // 채워진 부분의 색상
//             stroke: '#004080', // 외곽선 색상
//         },
//         style: {
//             shape: (x, y, r) => {
//                 const w = r * 2; // 전체 너비
//                 const h = r * 2; // 전체 높이

//                 // 사각형 모양의 경로 정의
//                 return [
//                     ['M', x - w / 2, y - h / 2], // 왼쪽 위
//                     ['L', x + w / 2, y - h / 2], // 오른쪽 위
//                     ['L', x + w / 2, y + h / 2], // 오른쪽 아래
//                     ['L', x - w / 2, y + h / 2], // 왼쪽 아래
//                     ['Z'], // 닫기
//                 ];
//             },
//             outlineBorder: 2,       // 외곽 두께
//             outlineDistance:6,
//             waveLength: 50,        // 파동
//             textFill: '#fff',       // 글자색
//         },
//     };

//     return (
//         <div className={`${size}`}>
//             <Liquid {...config} />
//         </div>
//     );
// }
