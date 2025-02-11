import { FaArrowRightLong } from "react-icons/fa6";
import React, { useEffect, useState } from 'react'
import InfoCard from "./InfoCard";

export default function DashWaterInfo({ data, predictionData }) {

    const [waterLevel, setWaterLevel] = useState("-");
    const [waterVol, setWaterVol] = useState("-");
    const [anticipated, setAnticipated] = useState("-");
    const [gap, setGap] = useState("");
    const [predctionHour, setPredictionHour] = useState("");

    const prediction = predictionData ? predictionData.data : null;

    useEffect(() => {
        // console.log("[DashWaterInfo] data :", data);
        // console.log("[DashWaterInfo] prediction :", prediction);
        if (!data) return;
        setWaterLevel(`${data.crtWaterHeight.toFixed(2)} / ${data.height}`);
        setWaterVol(`${data.waterVol.toFixed(1)} / ${data.capacity}`);

        if (!prediction) return;
        const anticipated = data.waterVol + data.input - prediction;
        setAnticipated(`${(anticipated).toFixed(1)}`);
        // console.log("predictionData****** => ", prediction);
        // console.log("****** => ", data.waterVol + data.input - prediction);
        setGap(`${(anticipated - data.waterVol.toFixed(1)) < 0 ? "▼" : "▲"}${(Math.abs(anticipated - data.waterVol.toFixed(1)).toFixed(1))}`)
        setPredictionHour(`${parseInt(predictionData.hour)+1}:00 기준 예측값`);

    }, [data, prediction]);


    return (
        <div className='w-full h-full rounded-lg grid grid-rows-3 gap-6'>
            <InfoCard color={"green"} label={"현재 수위 (m)"} value={waterLevel} detail={"현재 수위 / 최대 수위"} />
            <InfoCard color={"blue"} label={"현재 저수량 (m³)"} value={waterVol} detail={"현재 저수량 / 최대 용량"} />
            <InfoCard color={"pink"} label={"예상 저수량 (m³)"} value={<>{anticipated} <span className="text-lg"> {gap}</span></>}
                detail={predctionHour}
            />
        </div>
    )
}
