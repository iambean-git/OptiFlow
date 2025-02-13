import { BsEnvelopeCheck } from "react-icons/bs";
import React from 'react'

export default function CustomToast({ msg, closeToast }) {
    return (
        <div className="flex items-center w-full justify-between bg-[#f0f9f2] text-black p-4 rounded-lg  shadow-md ">
            <div className="text-green-600 text-4xl mr-4"><BsEnvelopeCheck/></div>
            <div className='flex flex-col items-start'>
                {
                    msg.map((item) => (
                        <span className="text-sm">{item}</span>
                    ))
                }
            </div>

            <button
                onClick={closeToast}
                className="ml-6 text-gray-400 hover:text-gray-600"
            >
                X
            </button>
        </div>
    );
}
