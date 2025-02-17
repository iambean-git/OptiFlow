import React, { useEffect, useRef, useState } from 'react';
import Chart from "react-apexcharts";


export default function WaterOutFlowGraph({ graphTitle, data, datepickerOption }) {
    const [chartXaxis, setChartXaxis] = useState([]);
    const [chartValue, setChartValue] = useState([]);
    const [chartValuePredict, setChartValuePredict] = useState([]);
    const [state, setState] = useState(null);

    useEffect(() => {
        if (!data) return;
        // console.log("🟡 [WaterOutFlowGraph] 유출량 데이터 :", data);

        setChartXaxis(data.date);
        setChartValue(data.output);
        setChartValuePredict(data.predict);
    }, [data]);

    useEffect(() => {
        if (!chartXaxis || !chartValue) return;

        const chartState = {
            series: [
                {
                    name: "실측값",
                    data: chartValue
                },
                {
                    name: "예측값",
                    data: chartValuePredict
                }
            ],
            options: {
                chart: {
                    type: 'line',
                    fontFamily: 'SUIT',
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
                            // 시간별 선택시
                            if (datepickerOption === "hourly") {
                                const splitValue = value.split('T'); // "2023-10-21T12:00:00" 형태 처리
                                if (splitValue.length === 2) {
                                    const timeValue = splitValue[1].substr(0, 2); // "12:00:00"에서 "12" 추출
                                    return timeValue; // X축 레이블에는 시간만 반환
                                }
                            }
                            // 일별 선택시
                            else if (datepickerOption === "daily") {
                                return value.substr(8, 2);
                            }
                            // 월별 선택시 
                            return value.substr(5, 2);
                        },
                    },
                    tooltip: {
                        enabled: false,
                    }
                },

                yaxis:
                {   
                    min: 0,
                    labels: {
                        formatter: function (value) {
                            return Math.round(value);  // Y축에서 소수점 제거
                        }
                    }
                },
                tooltip: {
                    x: {
                        formatter: function (value, { dataPointIndex }) {
                            const rawValue = chartXaxis[dataPointIndex]; // categories에서 원본 값 가져오기
                            const splitValue = rawValue.split('T');
                            if (splitValue.length === 2) {
                                const date = splitValue[0];
                                const time = parseInt(splitValue[1].substr(0, 2));
                                return `${date.replace(/-/g, "-")} ${time}시`; // 날짜와 시간 포맷팅
                            }
                            return rawValue;
                        },
                    },

                    y:
                        [
                            {
                                title: {
                                    formatter: function (val) {
                                        return val;
                                    }
                                },
                                formatter: function (value) {
                                    return value.toFixed(2) + " (m³)";
                                }
                            },
                            {
                                title: {
                                    formatter: function (val) {
                                        return val;
                                    }
                                },
                                formatter: function (value) {
                                    return value.toFixed(2) + " (m³)";
                                }
                            },
                        ]
                },
                grid: {
                    borderColor: '#f1f1f1',
                }
            },
        };

        setState(chartState);

    }, [chartXaxis, chartValue, chartValuePredict]);

    return (
        <div className='w-full h-full flex flex-col'>
            {
                state == null ?
                    <div> 데이터 로딩 중 </div>
                    : <Chart options={state.options} series={state.series} type="line" height={"93%"} />
            }
        </div>
    )
}
