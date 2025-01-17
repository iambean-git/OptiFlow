import "./flowStyles.css";

import { useState, useEffect } from 'react';
import { Handle, Position } from '@xyflow/react';

import WaterTest from "./WaterTest";

export default function CustomNode({ data, isConnectable }) {
    //비율 계산 (소수점 둘째자리까지만)
    const [theight, setTHeight] = useState("");

    useEffect(() => {
        if (data.capacity && data.crtVol) {
            console.log("data.capacity :", data.capacity);
            console.log("data :", data.crtVol);
            // setTHeight(`h-[${(data.crtVol / data.capacity * 10).toFixed(2)}vh]`);
            setTHeight(`${(data.crtVol / data.capacity * 10).toFixed(2)}vh`);
        }
    }, [data]);

    useEffect(()=>{
        if(!theight) return;
        console.log("theight:",theight);
    },[theight]);

    

    return (
        <>
            <Handle
                type="target"
                position={Position.Top}
                isConnectable={isConnectable}
            />
            <div className="w-24 h-32  flex flex-col items-center ">
                <div className="size-24 border border-t-0 border-gray-700 relative">
                    {/* <WaterTest percent={0.75} size={"size-24"} /> */}
                    <div style={{height : theight}}
                    className={`w-full bg-blue-600 absolute bottom-0`}> </div>
                </div>
                <p> {data.crtVol} / {data.capacity} </p>
            </div>

            <Handle
                type="source"
                position={Position.Bottom}
                id="a"
                isConnectable={isConnectable}
            />
        </>

    )
}
