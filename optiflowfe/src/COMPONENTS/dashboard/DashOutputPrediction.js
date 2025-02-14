import { GoDotFill } from "react-icons/go";
import { GoQuestion } from "react-icons/go";
import React, { useEffect, useState } from 'react';
import { Tooltip } from "react-tooltip";
import Chart from "react-apexcharts";

export default function DashOutputPrediction({ data, setModel }) {

    const [chartXaxis, setChartXaxis] = useState([]);
    const [chartValuePrediction, setChartValuePrediction] = useState([]);
    const [chartValueOptiflow, setChartValueOptiflow] = useState([]);
    const [chartValueHeight, setChartValueHeight] = useState([]);
    const [state, setState] = useState(null);
    // const [selected, setSelected] = useState("xgb");

    useEffect(() => {
        if (!data) return;
        // console.log("data" , data);
        setChartXaxis(data.time);
        setChartValuePrediction(data.prediction);
        setChartValueOptiflow(data.optiflow);
        setChartValueHeight(data.height);
    }, [data]);

    // useEffect(()=>{
    //     console.log("옵션 변경 : ",selected);
    // },[selected]);

    useEffect(() => {
        if (!chartXaxis || !chartValuePrediction) return;
        const chartState = {

            series: [
                {
                    name: "유출량 예측값",
                    data: chartValuePrediction
                },
                {
                    name: "추천 유입량",
                    data: chartValueOptiflow
                },
                {
                    name: "예상 수위",
                    data: chartValueHeight
                },
            ],
            options: {
                chart: {
                    height: 350,
                    type: 'line',
                    zoom: {
                        enabled: false
                    },
                    toolbar: {
                        tools: {
                            download: false,
                        },
                    },
                },
                dataLabels: {
                    enabled: false
                },
                stroke: {
                    width: 3,
                    curve: 'straight',
                },
                legend: {
                    // 툴팁 포매터 설정
                    tooltipHoverFormatter: function (val, opts) {
                        return val;
                    },
                    onItemClick: {
                        toggleDataSeries: false
                    },
                },
                markers: {
                    size: 4,
                    hover: {
                        sizeOffset: 2
                    }
                },
                xaxis: {
                    categories: chartXaxis,
                    labels: {
                        formatter: function (value) {
                            if (!value) return;
                            const splitValue = value.split('T'); // "2023-10-21T12:00:00" 형태 처리
                            if (splitValue.length === 2) {
                                const timeValue = splitValue[1].substr(0, 2); // "12:00:00"에서 "12" 추출
                                return timeValue; // X축 레이블에는 시간만 반환
                            }
                            return value;
                        },
                    },
                    tooltip: {
                        enabled: false,
                    }
                },

                yaxis: [
                    {
                        min: Math.min(...chartValuePrediction, ...chartValueOptiflow),  // 유출량 & 유입량 최소값
                        max: Math.max(...chartValuePrediction, ...chartValueOptiflow),  // 유출량 & 유입량 최대값
                        labels: {
                            formatter: function (value) {
                                return Math.round(value) + "";  // Y축에서 소수점 제거
                            }
                        },

                    },
                    {
                        labels: { show: false }, // 레이블 숨김
                        min: Math.min(...chartValuePrediction, ...chartValueOptiflow),  // 유출량 & 유입량 최소값
                        max: Math.max(...chartValuePrediction, ...chartValueOptiflow),  // 유출량 & 유입량 최대값

                    },
                    {
                        opposite: true,
                        min: 0,
                        max: 100,
                        labels: {
                            formatter: function (value) {
                                return Math.round(value);  // Y축에서 소수점 제거
                            }
                        },
                    }
                ],
                tooltip: {
                    x: {
                        formatter: function (value, { dataPointIndex }) {
                            const rawValue = chartXaxis[dataPointIndex]; // categories에서 원본 값 가져오기
                            const splitValue = rawValue.split('T');
                            if (splitValue.length === 2) {
                                const date = splitValue[0];
                                const time = parseInt(splitValue[1].substr(0, 2));
                                return `${date.replace(/-/g, "-")} ${time}~${time + 1}시`; // 날짜와 시간 포맷팅
                            }
                            return rawValue;
                        },
                    },

                    y:
                    [
                        {
                            title: {
                                formatter: function (val) {
                                    return val + " (m³)"
                                }
                            },
                            formatter: function (value) {
                                return value.toFixed(2) + " m³";  // 툴팁에서는 소수점 2자리까지 유지
                            },
                        },
                        {
                            title: {
                                formatter: function (val) {
                                    return val + " (m³)"
                                }
                            },
                            formatter: function (value) {
                                return value.toFixed(2) + " m³";  // 툴팁에서는 소수점 2자리까지 유지
                            },
                        },
                        {
                            title: {
                                formatter: function (val) {
                                    return val + " (%)"
                                }
                            },
                            formatter: function (value) {
                                return value.toFixed(2) + " %";  // 툴팁에서는 소수점 2자리까지 유지
                            },
                        },
                    ]
                },
                grid: {
                    borderColor: '#f1f1f1',
                }
            },
        };

        setState(chartState);


    }, [chartXaxis, chartValuePrediction]);


    return (
        <div className='w-full h-full p-6 flex flex-col'>

            {/* 제목 */}
            <div className='w-full flex justify-start items-center relative'>
                <span className="mr-2">유출량 예측값 및 추천 유입량</span>
                <GoQuestion className="text-gray-600" data-tooltip-id="detailtooltip" />
                <select className="absolute px-2 py-1 right-2 border border-[#5765b6] rounded-md text-sm text-[#5765b6] font-semibold focus:outline-none"
                    onClick={(e) => setModel(e.target.value)}
                >
                    <option value="xgb" className="text-sm text-[#333]">XG BOOST</option>
                    <option value="lstm" className="text-sm text-[#333]">LSTM</option>
                </select>
            </div>


            {/* 그래프 */}
            <div className='w-full flex-grow '>
                {
                    state == null ?
                        <div> 데이터 로딩 중 </div>
                        : <Chart options={state.options} series={state.series} type="line" height={"100%"} />
                }
            </div>

            <Tooltip
                id="detailtooltip"
                opacity={1}
                // className="!bg-[#779974] !z-10 !py-4 !px-6"
                className="!bg-gray-500 !z-10 !py-4 !px-6"

                // place="top-start"
                place="right"

            >
                <div className="flex flex-col mb-5 ">
                    <span className="text-base flex items-center"> <GoDotFill className="text-xs mr-1" /> 유출량 예측값 ?</span>
                    <span className="!text-sm ml-3"> 정각 기준, 향후 1시간 동안 예상되는 유출량 (m³)</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-base flex items-center"> <GoDotFill className="text-xs mr-1" /> 추천 유입량 ?</span>
                    <span className="!text-sm ml-3"> 안정 수위 유지와 전기 요금 절감을 위한 추천 유입량 (m³) </span>
                </div>
            </Tooltip>
        </div>
    )
}
