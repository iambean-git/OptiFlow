import React from 'react'
import { IoWaterOutline } from "react-icons/io5";
import { FaWater } from "react-icons/fa";
import { BsGraphUp } from "react-icons/bs";
export default function InfoCard({ color, label, value, detail }) {
    const textColors = { green: "text-[#4ba650]", blue: "text-[#3a93ee]", pink: "text-[#e53870]" };
    const bgColors = { green: "bg-[#4ba650]", blue: "bg-[#3a93ee]", pink: "bg-[#e53870]" };
    return (
        <div className='w-full h-full relative'>

            <div className={`size-12 absolute top-0 left-3 z-10 rounded-xl text-white ${bgColors[color]}
                            flex text-3xl justify-center items-center`}>
                {
                    color == "green" ? <FaWater /> :
                        color == "blue" ? <IoWaterOutline /> :
                            <img src='/images/dashboard/icon3.png' alt='icon3' className='size-9' />
                }

            </div>

            <div className='w-full h-[92%] px-5 bg-white absolute bottom-0 rounded-lg 
                            flex flex-col justify-center items-end'>
                <span className='text-gray-500 font-medium'>
                    {label}
                </span>

                <span className={`pt-0.5 text-[26px] font-bold ${textColors[color]}`}>
                    {value}
                </span>

                <span className='text-xs text-gray-500 mt-[-2px]'>
                    {detail}
                </span>
            </div>
        </div>
    )
}
