import React, { useEffect, useRef, useState } from 'react';
import Chart from "react-apexcharts";

export default function DashOutputPrediction({ data }) {

    // const [xaxis_category]
    useEffect(() => {
        if (!data) return;
        console.log("4️⃣ [DashOutputPrediction] data : ", data);

        const timeArray = [];
        const valueArray = [];

        // prediction.forEach(item => {
        //     timeArray.push(item.time);
        //     valueArray.push(item.value);
        // });
    }, [data]);

    const [state, setState] = useState({

        series: [{
            name: "Session Duration",
            data: [45, 52, 38, 24, 33, 26, 21, 20, 6, 8, 15, 10]
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
                    return val + ' - <strong>' + opts.w.globals.series[opts.seriesIndex][opts.dataPointIndex] + '</strong>'
                }
            },
            markers: {
                size: 0,
                hover: {
                    sizeOffset: 6
                }
            },
            xaxis: {
                categories: ['01 Jan', '02 Jan', '03 Jan', '04 Jan', '05 Jan', '06 Jan', '07 Jan', '08 Jan', '09 Jan',
                    '10 Jan', '11 Jan', '12 Jan'
                ],
            },

            yaxis: {
                labels: {
                    formatter: function (value) {
                        return Math.round(value);  // Y축에서 소수점 제거
                    }
                }
            },
            tooltip: {
                y: [
                    {
                        title: {
                            formatter: function (val) {
                                return val + " (mins)"
                            }
                        }
                    },
                    {
                        title: {
                            formatter: function (val) {
                                return val + " per session"
                            }
                        }
                    },
                    {
                        title: {
                            formatter: function (val) {
                                return val;
                            }
                        }
                    }
                ]
            },
            grid: {
                borderColor: '#f1f1f1',
            }
        },
    });

    return (
        <div className='w-full h-full p-6 flex flex-col'>

            {/* 제목 */}
            <div className='w-full '>유출량 예측값</div>

            {/* 그래프 */}
            <div className='w-full flex-grow '>
                <Chart options={state.options} series={state.series} type="line" height={"100%"} />
            </div>

        </div>
    )
}
