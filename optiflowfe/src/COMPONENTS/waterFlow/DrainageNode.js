import { useState, useEffect } from 'react';
import { Handle, Position } from '@xyflow/react';

export default function DrainageNode({ data }) {
    const [styleHeight, setStyleHeight] = useState("");
    const [crtVol, setCrtVol] = useState("");
    useEffect(() => {
        if (data.capacity && data.maxHeight && data.crtHeight) {


            console.log("data.capacity :", data.capacity);
            console.log("data.maxHeight :", data.maxHeight);
            console.log("data.crtHeight :", data.crtHeight);

            console.log("현재 수위 : ", (data.crtHeight / data.maxHeight * data.capacity).toFixed(1));
            // 현재 차있는 양
            setCrtVol((data.crtHeight / data.maxHeight * data.capacity).toFixed(1));
            // style 높이 지정 (소수 둘째자리까지)
            setStyleHeight(`${(data.crtHeight / data.maxHeight * 100).toFixed(1)}%`);
        }
    }, [data]);


    //styleHeight 콘솔 확인용
    useEffect(() => {
        if (!styleHeight) return;
        console.log("styleHeight:", styleHeight);
    }, [styleHeight]);


    return (
        <>
            <div className="w-20 h-28  flex flex-col items-center justify-end" >
                <div className='text-sm w-full h-6 text-center '> {data.label.substr(0, 1)} </div>

                <div className="w-20 h-16 border-2 border-t-0  p-0.5   rounded-b-md border-[#0c4296] relative ">
                    <div className='w-full h-full bg-[#81b0f9] bg-opacity-5' >
                        <div
                            style={{ height: styleHeight, maxHeight: "93%", width : "72px" }}
                            className={` bg-[#81b0f9] rounded-b-md absolute bottom-0.5`}> </div>
                        <div className='absolute w-full text-lg text-center top-4 text-[#0c4296] font-bold'>{styleHeight}</div>
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

            {/* <Handle
                type="source"
                position={Position.Bottom}
                id="b-source"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />

            <Handle
                type="target"
                position={Position.Bottom}
                id="b-target"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            /> */}

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
