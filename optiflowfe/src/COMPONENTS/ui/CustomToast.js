import { BsEnvelopeCheck } from "react-icons/bs";
import React from 'react'

export default function CustomToast({ type = "light", msg, closeToast }) {
    return (
        <div className={`flex items-center w-full justify-between text-black p-4 rounded-lg  shadow-md
                        ${type === "dark" ? "bg-[#4ba650]" : "bg-[#f0f9f2]"}`}>
            <div className={`text-4xl mr-4 ${type === "dark" ? "text-white" : "text-green-600"}`}>
                <BsEnvelopeCheck />
            </div>
            <div className='flex flex-col items-start'>
                {
                    msg.map((item, idx) => (
                        <span key={`toast-${idx}`}
                        className={`text-sm ${type === "dark" ? "text-white" : "text-[#333]" }`}>{item}</span>
                    ))
                }
            </div>

            <button
                onClick={closeToast}
                className={`ml-6 ${type === "dark" ? "text-white" : "text-gray-400 hover:text-gray-600" }`}
            >
                X
            </button>
        </div>
    );
}
