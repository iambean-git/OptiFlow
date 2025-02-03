import React, { useEffect, useRef, useState } from 'react';
import ApexCharts from 'apexcharts';

import testdata from "../../assets/data/testdata.json";

export default function WaterOutFlowGraph({ graphTitle, data, datepickerOption }) {

    // console.log("[WaterOutFlowGraph] data : ", data);
    // console.log("[WaterOutFlowGraph] 그래프타이틀 : ", graphTitle);
    // console.log("[WaterOutFlowGraph] datepickerOption : ", datepickerOption);
    const dateUnit = { hourly: "시", daily: "일", monthly: "월" };
    const axisX = Object.keys(data);
    const data1 = Object.values(data);
    const data2 = Object.values(data);
    const chartRef = useRef(null);

    const options = {
        series: [{
            name: "실측값",
            data: data1
        },
        {
            name: "예측값",
            data: data2
        },
        ],
        chart: {
            width: '660px',
            height: '280px',
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
            width: [3, 3],
            curve: 'straight',
            // dashArray: [0, 8, 5]
        },
        title: {
            text: `${graphTitle} 배수지 시간별 유출량 비교`,
            align: 'left',
            style : {
                fontFamily: 'SUIT',
            }
        },
        legend: {
            tooltipHoverFormatter: function (val, opts) {
                return val;
            }

        },
        markers: {
            size: 0,
            hover: {
                sizeOffset: 6
            }
        },
        xaxis: {
            categories: axisX,
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

                formatter: function (val) {
                    return val + dateUnit[datepickerOption]
                },
            },

            y: [
                {
                    formatter: function (value) {
                        return value.toFixed(2);  // 툴팁에서는 소수점 2자리까지 유지
                    },
                    
                    title: {
                        formatter: function (val) {
                            return val + " (m³)"
                        }
                    }
                },
                {
                    title: {
                        formatter: function (val) {
                            return val + " (m³)"
                        }
                    }
                },
            ],
        },

        grid: {
            borderColor: '#f1f1f1',
        }
    };

    useEffect(() => {
        const chart = new ApexCharts(chartRef.current, options);
        chart.render();

        return () => {
            chart.destroy(); // Clean up the chart on component unmount
        };
    }, [graphTitle, datepickerOption, data]);

    return (
        <div>
            <div id="chart" ref={chartRef} className=""></div>
        </div>
    )
}
