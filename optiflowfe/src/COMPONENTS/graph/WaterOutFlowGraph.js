import React, { useEffect, useRef, useState } from 'react';
import ApexCharts from 'apexcharts';

import testdata from "../../assets/data/testdata.json";

export default function WaterOutFlowGraph({ graphTitle, data }) {
    
    console.log("[WaterOutFlowGraph] 그래프타이틀 : ", graphTitle);

    const axisX = Object.keys(testdata.hourly_water_outflow);
    const data1 = Object.values(testdata.hourly_water_outflow);
    const data2 = Object.values(testdata.predicted_water_outflow);
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
            // {
            //     name: 'Total Visits',
            //     data: [87, 57, 74, 99, 75, 38, 62, 47, 82, 56, 45, 47]
            // }
        ],
        chart: {
            width: '560px',
            height: '300px',
            type: 'line',
            zoom: {
                enabled: false
            },
            toolbar: {
                tools: {
                    download: false,
                    // csv: false,         //csv 다운로드 비활성화 (안먹힘)
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
            text: `${graphTitle} 시간별 유출량 비교`,
            align: 'left'
        },
        legend: {
            // customLegendItems: ["실측값", "예측값"],
            tooltipHoverFormatter: function (val, opts) {
                return val ;
            }
            // onItemClick: {
            //     toggleDataSeries: true // 기본값: true
            // }
        },
        markers: {
            size: 0,
            hover: {
                sizeOffset: 6
            }
        },
        xaxis: {
            // categories: ['01 Jan', '02 Jan', '03 Jan', '04 Jan', '05 Jan', '06 Jan', '07 Jan', '08 Jan', '09 Jan',
            //     '10 Jan', '11 Jan', '12 Jan'
            // ],

            categories: axisX,
            tooltip :{
                enabled:false,
            }
        },
        tooltip: {
            x: {
                
                formatter: function (val) {
                    return val + "시"
                },
            },

            y: [
                {
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
    }, [graphTitle]);

    return (
        <div>
            <div id="chart" ref={chartRef} className=""></div>
        </div>
    )
}
