import { FaArrowRightLong } from "react-icons/fa6";
import React, { useEffect, useState } from 'react'
import InfoCard from "./InfoCard";

export default function DashWaterInfo({data}) {
    
    const[waterLevel, setWaterLevel] = useState("-");
    const[waterVol, setWaterVol] = useState("-");
    const[anticipated, setAnticipated] = useState("-");
    const[gap, setGap] = useState("-");

    useEffect(()=>{
        if(!data)   return;
        console.log("[DashWaterInfo] data :", data);

        setWaterLevel(`${data.crtWaterHeight.toFixed(2)} / ${data.height}`);
        setWaterVol(`${data.waterVol.toFixed(1)} / ${data.capacity}`);
        setAnticipated(`1123`);
        setGap(`${(1123-data.waterVol.toFixed(1))<0 ? "▼" : "▲"}${(Math.abs(1123-data.waterVol.toFixed(1)).toFixed(1))}`)
    },[data]);

    return (
        <div className='w-full h-full rounded-lg grid grid-rows-3 gap-6'>
            <InfoCard color={"green"} label={"현재 수위 (m)"} value={waterLevel} detail = {"현재 수위 / 최대 수위"}/>
            <InfoCard color={"blue"} label={"현재 저수량 (m³)"} value={waterVol} detail = {"현재 저수량 / 최대 용량"}/>
            <InfoCard color={"pink"} label={"예상 저수량 (m³)"} value={<>{anticipated} <span className="text-lg"> {gap}</span></>} detail = {"현재 시간 기준 1시간 후"}/>
        </div>
    )
}
