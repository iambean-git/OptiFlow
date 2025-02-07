import { useState, useEffect } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Tooltip } from "react-tooltip";

export default function DrainageNode({ data }) {
    const [heightText, setHeightText] = useState("0%");
    const [crtVol, setCrtVol] = useState("");
    useEffect(() => {
        if (data.capacity && data.maxHeight && data.crtHeight) {
            setCrtVol((data.crtHeight / data.maxHeight * data.capacity).toFixed(1));
            setHeightText(`${(data.crtHeight / data.maxHeight * 100).toFixed(1)}%`); // 텍스트용 높이 지정 (소수 둘째자리까지)
        }
    }, [data]);

    return (
        <>
            <div className="w-20 h-28  flex flex-col items-center justify-end"
                data-tooltip-id="my-tooltip">
                <div className='text-sm w-full h-6 text-center '> {data.label.substr(0, 1)} </div>

                <div className="w-20 h-16 border-2 border-t-0  p-0.5   rounded-b-md border-[#0c4296] relative ">
                    <div className='w-full h-full bg-[#81b0f9] bg-opacity-5' >
                        <div
                            style={{
                                height: heightText, maxHeight: "93%", width: "72px",
                                transition: "height 0.8s ease-in-out"
                            }}
                            className={` bg-[#81b0f9] rounded-b-md absolute bottom-0.5`}> </div>
                        <div className='absolute w-full text-lg text-center top-4 text-[#0c4296] font-bold'>{heightText}</div>
                    </div>

                </div>
                <p className='text-xs'> {crtVol} / {data.capacity} </p>
            </div>

            <Tooltip
                id="my-tooltip"
                className="!bg-gray-300"
                style={{
                    // backgroundColor: "#facc15", // Tailwind bg-yellow-200
                    color: "#374151", // Tailwind text-gray-300 (대신 #374151로 더 가시성 높임)
                    fontSize: "14px",
                    padding: "8px 12px",
                    borderRadius: "8px",
                }}
            >
                <div className="flex flex-col">
                    <div>J 정수장 </div>
                    <div>수위 28/70</div>
                    <div>저수량 2812/3000</div>
                    <div>안정 저수량 900~2850 </div>
                </div>
            </Tooltip>

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
