import React, { useEffect, useState } from 'react';
import Chart from "react-apexcharts";

export default function DashOutput({ data }) {
    const [chartXaxis, setChartXaxis] = useState([]);
    const [chartValueOutput, setChartValueOutput] = useState([]);
    const [chartValueInput, setChartValueInput] = useState([]);
    const [chartValueHeight, setChartValueHeight] = useState([]);
    const [state, setState] = useState(null);

    useEffect(() => {
        if (!data) return;
        // console.log("data" , data);
        setChartXaxis(data.time);
        setChartValueOutput(data.output);
        setChartValueInput(data.input);
        setChartValueHeight(data.height);
    }, [data]);

    useEffect(() => {
        if (!chartXaxis || !chartValueOutput) return;
        const chartState = {

            series: [
                {
                    name: "유출량",
                    data: chartValueOutput
                },
                {
                    name: "유입량",
                    data: chartValueInput
                },
                {
                    name: "수위",
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
                        min: Math.min(...chartValueOutput, ...chartValueInput),  // 유출량 & 유입량 최소값
                        max: Math.max(...chartValueOutput, ...chartValueInput),  // 유출량 & 유입량 최대값
                        labels: {
                            formatter: function (value) {
                                return Math.round(value)+"";  // Y축에서 소수점 제거
                            }
                        },

                    },
                    {
                        labels: { show: false }, // 레이블 숨김
                        min: Math.min(...chartValueOutput, ...chartValueInput),  // 유출량 & 유입량 최소값
                        max: Math.max(...chartValueOutput, ...chartValueInput),  // 유출량 & 유입량 최대값

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
                                // return `${date.replace(/-/g, "-")} ${time}~${time + 1}시`; // 날짜와 시간 포맷팅
                                return `2024-10-17 ${time}~${time + 1}시`; // 날짜와 시간 포맷팅
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

    }, [chartXaxis, chartValueOutput]);

    return (
        <div className='w-full h-full p-6 flex flex-col'>

            {/* 제목 */}
            <div className='w-full flex justify-between items-end '>
                <span>이전 유출/유입량 및 수위</span>
            </div>

            {/* 그래프 */}
            <div className='w-full flex-grow '>
                {
                    state == null ?
                        <div> 데이터 로딩 중 </div>
                        : <Chart options={state.options} series={state.series} type="line" height={"100%"} />
                }
            </div>

        </div>
    )
}
