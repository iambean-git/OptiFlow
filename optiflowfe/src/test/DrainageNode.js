import { useState, useEffect } from 'react';
import { Handle, Position } from '@xyflow/react';

export default function DrainageNode({ data, isConnectable, dragHandle }) {
    const [vhHeight, setVhHeight] = useState("");

    useEffect(() => {
        if (data.capacity && data.crtVol) {
            console.log("data.capacity :", data.capacity);
            console.log("data :", data.crtVol);

            //vh높이 지정 (소수 둘째자리까지)
            setVhHeight(`${(data.crtVol / data.capacity * 100).toFixed(1)}%`);
        }
    }, [data]);


    //vhHeight 콘솔 확인용
    useEffect(() => {
        if (!vhHeight) return;
        console.log("vhHeight:", vhHeight);
    }, [vhHeight]);

    return (
        <>

            <div className="w-20 h-20  flex flex-col items-center ">
                <div className="w-20 h-16 border border-t-0 border-gray-700 relative">
                    {/* <WaterTest percent={0.75} size={"size-24"} /> */}
                    <div
                        style={{ height: vhHeight }}
                        // style={{height:"50.2%"}}
                        className={`w-full h-[10vh] bg-blue-600 absolute bottom-0`}> </div>
                    <div className='absolute  text-[10px] top-4 left-2'>{data.label}({vhHeight})</div>
                </div>
                <p className='text-xs'> {data.crtVol} / {data.capacity} </p>
            </div>


            <Handle
                type="source"
                position={Position.Top}
                id="t-source"
                className="w-16 !bg-teal-500"
            />

            <Handle
                type="target"
                position={Position.Top}
                id="t-target"
                className="w-16 !bg-teal-500"
            />

            <Handle
                type="source"
                position={Position.Bottom}
                id="b-source"
                className="w-16 !bg-teal-500"
            />

            <Handle
                type="target"
                position={Position.Bottom}
                id="b-target"
                className="w-16 !bg-teal-500"
            />

            <Handle
                type="source"
                position={Position.Right}
                id="r-source"
                className="w-16 !bg-teal-500"
            />
            <Handle
                type="target"
                position={Position.Right}
                id="r-target"
                className="w-16 !bg-teal-500"
            />

            <Handle
                type="source"
                position={Position.Left}
                id="l-source"
                className="w-16 !bg-teal-500"
            />
            <Handle
                type="target"
                position={Position.Left}
                id="l-target"
                className="w-16 !bg-teal-500"
            />

        </>
    )
}
