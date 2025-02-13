import { FaArrowRightLong } from "react-icons/fa6";
import React, { useEffect, useRef, useState } from 'react';
import { Link } from "react-router-dom";
import Chart from "react-apexcharts";
import "./dashboard.css";

export default function DashWaterLevel({ data, selected, setSelected }) {
    const [state, setState] = useState(null);

    useEffect(() => {
        if (!data) return;
        console.log("🔥[DashWaterLevel] 렌더링 : ", data);

        const radialOptions = {};
        data.map((item) => {
            radialOptions[(item.id).toUpperCase()] = {
                series: [item.percentage],
                options: {
                    chart: {
                        type: 'radialBar',
                        offsetY: 13,  // y축 패딩
                    },
                    plotOptions: {
                        radialBar: {

                            hollow: {
                                size: "55%",  // 중앙 비어 있는 원 크기 조절
                                margin: 0,
                                background: 'white',
                            },
                            track: {
                                strokeWidth: "100%", // 배경 원의 두께 조절
                                // background: '#249efa',
                            },
                            dataLabels: {
                                name: {offsetY: -5},
                                value: {offsetY: 2},
                            },
                        },
                    },
                    grid: {
                        "padding": {
                            "top": -20,
                            "right": -10,
                            // "bottom": -10,
                            "left": -10
                        }
                    },
                    states: {
                        hover: {
                            filter: {
                                type: 'none',
                            },
                        },
                        active: {
                            filter: {
                                type: 'none',
                            }
                        }
                    },
                    stroke: {
                        lineCap: 'round',
                    },

                    colors: (item.percentage <= 30 || item.percentage >= 95) ? ["#E82F2E"]
                        : (item.percentage <= 40 || item.percentage >= 80) ? ["#ff9800"] : [],
                    labels: [(item.id).toUpperCase()],
                },
            }
        });

        console.log("🔥[DashWaterLevel] radialOptions : ", radialOptions);

        setState(radialOptions);
    }, [data]);

    useEffect(() => {
        if (!state) return;
        console.log("🔥[DashWaterLevel] state : ", state);
    }, [state]);

    const handleClick = (event) => {
        console.log(" 차트 클릭 : ", event.currentTarget.id);
        setSelected({ value: event.currentTarget.id, label: event.currentTarget.id + " 배수지" });
    };

    return (
        <div className='w-full h-full rounded-lg p-6 flex flex-col'>
            <div className='w-full flex justify-between items-center'>
                <span className="text-lg h-full">
                    배수지별 현재 저수량
                </span>

                <div className="flex items-center hover:border-b ">
                    <Link to="/waterlevel" className="text-sm text-gray-400">
                        이전 저수량 보기
                    </Link>
                    <span>
                        <FaArrowRightLong className="ml-2 text-gray-400" />
                    </span>

                </div>
            </div>
            <div className='w-full flex-grow grid grid-cols-6 gap-2 items-center'>
                {
                    state ?
                        <>
                            <Chart id="A" options={state.A.options} series={state.A.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "A" ? " selectedChart" : "unselectedChart"} radialChart `}
                                onClick={handleClick}
                            />

                            <Chart id="B" options={state.B.options} series={state.B.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "B" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                            <Chart id="C" options={state.C.options} series={state.C.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "C" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                            <Chart id="D" options={state.D.options} series={state.D.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "D" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                            <Chart id="E" options={state.E.options} series={state.E.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "E" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                            <Chart id="F" options={state.F.options} series={state.F.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "F" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />


                            <Chart id="G" options={state.G.options} series={state.G.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "G" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                            <Chart id="H" options={state.H.options} series={state.H.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "H" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                            <Chart id="I" options={state.I.options} series={state.I.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "I" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                            <Chart id="J" options={state.J.options} series={state.J.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "J" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                            <Chart id="K" options={state.K.options} series={state.K.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "K" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                            <Chart id="L" options={state.L.options} series={state.L.series} type="radialBar" width={"100%"}
                                className={`${selected.value == "L" ? " selectedChart" : "unselectedChart"}  radialChart `}
                                onClick={handleClick}
                            />
                        </>

                        :
                        <div> 데이터 로드 중 </div>
                }
            </div>


        </div>
    )
}
