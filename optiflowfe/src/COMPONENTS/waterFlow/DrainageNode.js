import { useState, useEffect } from 'react';
import { Handle, Position } from '@xyflow/react';

export default function DrainageNode({ data }) {

    // console.log(" 저수지 노드 data : ", data);
    const [animatedHeight, setAnimatedHeight] = useState("0%"); // 현재 애니메이션 높이
    const [heightText, setHeightText] = useState("0%");
    const [crtVol, setCrtVol] = useState("");
    const [animated, setAnimated] = useState()
    useEffect(() => {
        // if (data.capacity && data.maxHeight && data.crtHeight) {
        //     // console.log("data.capacity :", data.capacity);
        //     // console.log("data.maxHeight :", data.maxHeight);
        //     // console.log("data.crtHeight :", data.crtHeight);

        //     // console.log("현재 수위 : ", (data.crtHeight / data.maxHeight * data.capacity).toFixed(1));
        //     // 현재 차있는 양

        //     // 1. 물 높이를 먼저 0%로 초기화
        //     setAnimatedHeight("0%");

        //     // 2. 계산 진행
        //     setCrtVol((data.crtHeight / data.maxHeight * data.capacity).toFixed(1));
        //     setHeightText(`${(data.crtHeight / data.maxHeight * 100).toFixed(1)}%`); // 텍스트용 높이 지정 (소수 둘째자리까지)

        //     // 3. 약간의 딜레이 후 새로운 높이로 설정
        //     const timer = setTimeout(() => {
        //         setAnimatedHeight(`${(data.crtHeight / data.maxHeight * 100).toFixed(1)}%`);
        //         setAnimated("height 0.8s ease-in-out");
        //     }, 100); // 100ms 딜레이
        //     setAnimated("");
        //     return () => clearTimeout(timer); // 컴포넌트 언마운트 시 타이머 정리
        // }

        if (data.capacity && data.maxHeight && data.crtHeight) {

            // 2. 계산 진행
            setCrtVol((data.crtHeight / data.maxHeight * data.capacity).toFixed(1));
            setHeightText(`${(data.crtHeight / data.maxHeight * 100).toFixed(1)}%`); // 텍스트용 높이 지정 (소수 둘째자리까지)

            // 3. 약간의 딜레이 후 새로운 높이로 설정

            setAnimatedHeight(`${(data.crtHeight / data.maxHeight * 100).toFixed(1)}%`);

        }
    }, [data]);





    return (
        <>
            <div className="w-20 h-28  flex flex-col items-center justify-end" >
                <div className='text-sm w-full h-6 text-center '> {data.label.substr(0, 1)} </div>

                <div className="w-20 h-16 border-2 border-t-0  p-0.5   rounded-b-md border-[#0c4296] relative ">
                    <div className='w-full h-full bg-[#81b0f9] bg-opacity-5' >
                        <div
                            style={{ height: heightText, maxHeight: "93%", width: "72px", transition: "height 0.8s ease-in-out" }}
                            className={` bg-[#81b0f9] rounded-b-md absolute bottom-0.5`}> </div>
                        <div className='absolute w-full text-lg text-center top-4 text-[#0c4296] font-bold'>{heightText}</div>
                    </div>

                </div>
                <p className='text-xs'> {crtVol} / {data.capacity} </p>
            </div>


            <Handle
                type="source"
                position={Position.Top}
                id="t-source"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />

            <Handle
                type="target"
                position={Position.Top}
                id="t-target"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />

            <Handle
                type="source"
                position={Position.Right}
                id="r-source"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />
            <Handle
                type="target"
                position={Position.Right}
                id="r-target"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />

            <Handle
                type="source"
                position={Position.Left}
                id="l-source"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />
            <Handle
                type="target"
                position={Position.Left}
                id="l-target"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />

        </>
    )
}
