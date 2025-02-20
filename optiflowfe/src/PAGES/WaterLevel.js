import NavBar from "../components/NavBar";
import WaterFlow from "../components/waterFlow/WaterFlow";

import React, { useEffect, useState } from "react";

import "react-datepicker/dist/react-datepicker.css";
import DateNTime from "../components/datepicker/DateNTime";
import { formatDate } from "../utils/dateUtils";

import { maxDate10am } from "../recoil/DateAtom";
import { useRecoilValue } from "recoil";
import LoadingSpinner from "../components/ui/LoadingSpinner";
export default function WaterLevel() {
    const server = process.env.REACT_APP_SERVER_ADDR;

    const defaultDate = useRecoilValue(maxDate10am);
    const [selectedDate, setSelectedDate] = useState(defaultDate);
    const [waterFlowTag, setWaterFlowTag] = useState(LoadingSpinner);

    const [waterLevel, setWaterLevel] = useState('');

    useEffect(() => {
        if (!selectedDate) return;
        // console.log("💥 formatDate 확인 : ", formatDate(selectedDate));
        const fetchWaterLevelData = async () => {
            const url = `${server}/api/reservoirdata/${formatDate(selectedDate)}`;
            const resp = await fetch(url);
            const data = await resp.json();
            // console.log("🌊 [WaterLevel] 수위 데이터 :", data);

            // 동일한 값이면 업데이트 방지
            if (JSON.stringify(data) === JSON.stringify(waterLevel)) {
                console.log("⚠️ [WaterLevel] 동일한 수위 데이터, 업데이트 안함.");
                return;
            }
            setWaterLevel(data);
        };

        fetchWaterLevelData();
    }, [selectedDate]);


    useEffect(() => {
        if (!waterLevel) return;

        setWaterFlowTag(< WaterFlow waterLevel={waterLevel} />);
    }, [waterLevel]);

    return (
        <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2] ">
            <NavBar />
            <div className="w-full h-screen pl-[260px] flex flex-col">
                <div className="w-full h-[160px] px-10 flex justify-between">
                    {/* 텍스트 */}
                    <div className="w-4/5 h-full  flex flex-col justify-end text-[#333333]">
                        <h1 className="text-4xl font-medium">지난 수위 정보</h1>
                        <p className="mt-2">날짜를 변경하세요. <span className="whitespace-nowrap"> 각 배수지에 마우스를 올리면, 세부 정보도 확인할 수 있습니다.</span> </p>
                        <p className=""></p>
                    </div>

                    {/* 달력 */}
                    <div className="h-full  relative min-w-72 ">
                        <section className="absolute bottom-0 right-0 ">
                            <DateNTime selectedDate={selectedDate} setSelectedDate={setSelectedDate} />
                        </section>
                    </div>
                </div>
                <section className="px-10 pb-10 pt-6 w-full h-full">
                    <div className="w-full h-full border rounded-lg ">
                        {waterFlowTag}
                    </div>
                </section>
            </div>
        </div>
    );
}
