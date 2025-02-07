import React, { useEffect, useRef, useState } from 'react';
import Chart from "react-apexcharts";

export default function DashOutput({ data }) {
    const [chartXaxis, setChartXaxis] = useState([]);
    const [chartValue, setChartValue] = useState([]);
    const [chartValue2, setChartValue2] = useState([]);
    const [state, setState] = useState(null);

    useEffect(() => {
        if (!data) return;
        // console.log("data" , data);
        setChartXaxis(data.time);
        setChartValue(data.input);
        setChartValue2(data.output);
    }, [data]);

    useEffect(() => {
        if (!chartXaxis || !chartValue) return;
        const chartState = {

            series: [
                {
                    name: "유입량",
                    data: chartValue
                },
                {
                    name: "유출량",
                    data: chartValue2
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
                title: {
                    text: '',
                    align: 'left',

                },
                legend: {
                    // 툴팁 포매터 설정
                    tooltipHoverFormatter: function (val, opts) {
                        // return val + ' - <strong>' + opts.w.globals.series[opts.seriesIndex][opts.dataPointIndex] + '</strong>'
                        return val;
                    }
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

                yaxis: {
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
                                return `${date.replace(/-/g, "-")} ${time}~${time + 1}시`; // 날짜와 시간 포맷팅
                            }
                            return rawValue;
                        },
                    },

                    y: 
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
                },
                grid: {
                    borderColor: '#f1f1f1',
                }
            },
        };

        setState(chartState);

    }, [chartXaxis, chartValue]);

    return (
        <div className='w-full h-full p-6 flex flex-col'>

            {/* 제목 */}
            <div className='w-full flex justify-between items-end '>
                <span>이전 유입량 및 유출량</span>
                <span className='text-sm text-gray-500'>설명</span>
            </div>

            {/* 그래프 */}
            <div className='w-full flex-grow '>
                {
                    state == null ?
                        <div> 데이터 로딩 중 </div>
                        : <Chart options={state.options} series={state.series} type="line" height={"100%"} />
                }
            </div>
            {/* <div className='w-full flex justify-end items-end '>
                <span className='text-sm text-gray-500'>정각 기준, 향후 1시간 동안 예상되는 유출량 (m³)</span>
            </div> */}

        </div>
    )
}
