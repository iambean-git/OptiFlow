import React, { useEffect, useRef } from 'react';
import ApexCharts from 'apexcharts';

export default function GraphTest() {
    const chartRef = useRef(null);

    useEffect(() => {
        const options = {
            series: [{
                name: 'Website Blog',
                type: 'column',
                data: [440, 505, 414, 671, 227, 413, 201, 352, 752, 320, 257, 160]
              }, {
                name: 'Social Media',
                type: 'line',
                data: [23, 42, 35, 27, 43, 22, 17, 31, 22, 22, 12, 16]
              }],
                chart: {
                height: 350,
                type: 'line',
                toolbar : {
                    tools : {
                        download: true,     
                        // csv: false,         //csv 다운로드 비활성화 (안먹힘)
                    },
                    
                }
              },

              // 선두께
              stroke: {
                width: [0, 4]
              },

              // 그래프 타이틀
              title: {
                text: 'Traffic Sources'
              },
              dataLabels: {
                enabled: true,
                enabledOnSeries: [1]
              },
              
              labels: ['01 Jan 2001', '02 Jan 2001', '03 Jan 2001', '04 Jan 2001', '05 Jan 2001', '06 Jan 2001', '07 Jan 2001', '08 Jan 2001', '09 Jan 2001', '10 Jan 2001', '11 Jan 2001', '12 Jan 2001'],
              yaxis: [{
                title: {
                  text: 'Website Blog',
                },
              
              }, {
                opposite: true,
                title: {
                  text: 'Social Media'
                }
              }]
        };

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
