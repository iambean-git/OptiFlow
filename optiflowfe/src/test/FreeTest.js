import React, { useEffect, useRef } from 'react';
import ApexCharts from 'apexcharts';

import testdata from "../assets/data/testdata.json";

export default function FreeTest() {
    console.log("testData : ",Object.keys(testdata.hourly_water_outflow));
    const axisX = Object.keys(testdata.hourly_water_outflow);
    const data1 = Object.values(testdata.hourly_water_outflow);
    const data2 = Object.values(testdata.predicted_water_outflow);
    const chartRef = useRef(null);

    const options = {
        series: [{
            name: Object.keys(testdata)[0],
            data: data1
        },
        {
            name: Object.keys(testdata)[1],
            data: data2
        },
        // {
        //     name: 'Total Visits',
        //     data: [87, 57, 74, 99, 75, 38, 62, 47, 82, 56, 45, 47]
        // }
        ],
        chart: {
            height: 350,
            type: 'line',
            zoom: {
                enabled: false
            },
            toolbar : {
                tools : {
                    download: false,     
                    // csv: false,         //csv 다운로드 비활성화 (안먹힘)
                },
                
            }
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
            text: '유출량',
            align: 'left'
        },
        legend: {
            customLegendItems: [1,2],
            // tooltipHoverFormatter: function (val, opts) {
            //     return val + ' - <strong>' + opts.w.globals.series[opts.seriesIndex][opts.dataPointIndex] + '</strong>'
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
        },
        tooltip: {
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
    };

    useEffect(() => {
        const chart = new ApexCharts(chartRef.current, options);
        chart.render();

        return () => {
            chart.destroy(); // Clean up the chart on component unmount
        };
    }, []);

    return (
        <div>
            <div id="chart" ref={chartRef} className="max-w-[650px] m-9"></div>
        </div>
    );
}
