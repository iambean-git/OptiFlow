import { useState, useEffect } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Tooltip } from "react-tooltip";
import ReactDOM from "react-dom";
import "./tooltip.css";

export default function DrainageNode({ id, data }) {
    const [heightText, setHeightText] = useState("0%");
    const [crtVol, setCrtVol] = useState("");

    useEffect(() => {
        if (data.capacity && data.maxHeight && data.crtHeight) {
            setCrtVol((data.crtHeight / data.maxHeight * data.capacity).toFixed(1));
            setHeightText(`${(data.crtHeight / data.maxHeight * 100).toFixed(1)}%`);
        }
    }, [data]);

    return (
        <>
            <div
                className="w-20 h-28 flex flex-col items-center justify-end relative z-[50]"
                data-tooltip-id={`tooltip-${id}`}
            >
                <div className='text-sm w-full h-6 text-center '>
                    {data.label.substr(0, 1)}
                </div>

                <div className="w-20 h-16 border-2 border-t-0 p-0.5 rounded-b-md border-[#0c4296] relative ">
                    <div className='w-full h-full bg-[#81b0f9] bg-opacity-5'>
                        <div
                            style={{
                                height: heightText,
                                maxHeight: "93%",
                                width: "72px",
                                transition: "height 0.8s ease-in-out"
                            }}
                            className="bg-[#81b0f9] rounded-b-md absolute bottom-0.5"
                        />
                        <div className='absolute w-full text-lg text-center top-4 text-[#0c4296] font-bold'>
                            {heightText}
                        </div>
                    </div>
                </div>

                <p className='text-xs'> {crtVol} / {data.capacity} </p>
            </div>

            {
                data ? ReactDOM.createPortal(
                    <Tooltip
                        id={`tooltip-${id}`}
                        opacity={1}             //투명도
                        className="!bg-gray-300 !text-gray-700 !text-sm !px-3 !py-2 !rounded-lg z-10"
                        // openOnClick          // 클릭하면 나오게 하기
                        place={id=="drainJ"||id=="drainF" ?  "bottom" : "top"}
                    >
                        <div className="flex flex-col">
                            <div className='font-semibold text-lg text-[#0c4296]'>{data.label}</div>
                            <section className='flex'>
                                <div className='flex flex-col mr-3 text-[#0c4296]'>
                                    <div>수위(m)</div><div>저수량(m³)</div><div>안정 범위(m³)</div>
                                </div>
                                <div className='flex flex-col'>
                                    <div>{data.crtHeight.toFixed(1)} / {data.maxHeight}</div>
                                    <div>{crtVol} / {data.capacity}</div>
                                    <div>{data.capacity * 0.4} ~ {data.capacity * 0.8}</div>
                                </div>
                            </section>
                        </div>
                    </Tooltip>,
                    document.body // 툴팁을 body 안으로 완전히 분리!
                ) : null
            }

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
    );
}
