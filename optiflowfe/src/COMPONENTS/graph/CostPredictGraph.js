import React, { useEffect, useState } from 'react';
import Chart from "react-apexcharts";

export default function CostPredictGraph({ data, datepickerOption }) {
    const [chartXaxis, setChartXaxis] = useState([]);
    const [chartValueTruth, setChartValueTruth] = useState([]);
    const [chartValueOptimization, setChartValueOptimization] = useState([]);
    const [state, setState] = useState(null);

    useEffect(() => {
        if (!data) return;
        // console.log("🟡 [CostPredictGraph] 전기요금 데이터 :", data);

        setChartXaxis(data.date);
        setChartValueTruth(data.truth);
        setChartValueOptimization(data.optimization);
    }, [data]);

    useEffect(() => {
        if (!chartXaxis || !chartValueTruth) return;

        const chartState = {
            series: [
                {
                    name: "실 전기 요금",
                    data: chartValueTruth
                },
                {
                    name: "예측 전기 요금",
                    data: chartValueOptimization
                }
            ],
            options: {
                chart: {
                    height: 350,
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
                    labels: {
                        formatter: function (value) {
                            return new Intl.NumberFormat("ko-KR").format(value);
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
                                    return new Intl.NumberFormat("ko-KR").format(value) + "원";
                                }
                            },
                            {
                                title: {
                                    formatter: function (val) {
                                        return val;
                                    }
                                },
                                formatter: function (value) {
                                    return new Intl.NumberFormat("ko-KR").format(value) + "원";
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

    }, [chartXaxis, chartValueTruth, chartValueOptimization]);

    return (
        <div className='w-full h-full flex flex-col'>
            {
                state == null ?
                    <div> 데이터 로딩 중 </div>
                    : <Chart options={state.options} series={state.series} type="line" height={"100%"} />
            }
            {/* <div className='w-full flex justify-between'>
                <span>{graphTitle} 배수지 {dateOption[datepickerOption]} 유출량 비교</span>
                <span>{(Number(data?.percent) || 0).toFixed(2)}% 감소</span>
            </div> */}

        </div>
    )
}
