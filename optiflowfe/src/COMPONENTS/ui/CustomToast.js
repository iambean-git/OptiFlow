import React from 'react'

export default function CustomToast({ msg, closeToast }) {
    return (
        <div className="flex items-center justify-between bg-white text-black p-4 rounded-lg  border shadow-md max-w-xs">
            <span className="text-sm">{msg}</span>
            <button
                onClick={closeToast}
                className="ml-4 bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded text-xs"
            >
                닫기
            </button>
        </div>
    );
}
